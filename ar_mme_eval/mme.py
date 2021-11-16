import math
# import warnings
# warnings.filterwarnings('error')
import pandas as pd
from matplotlib.pylab import plt
from pandas.core.frame import DataFrame
from prompt_toolkit.shortcuts import set_title
import numpy as np

def eval(gt, pt,meta=None, clas=None,debug=0,calcne=0,args={}):
    #%matplotlib inline
    if clas is None:
        clas=gt.event_label.append(pt.event_label).unique()
    
    result={}
    total_time=0
    import time

    for c in clas:
        gtc=gt.loc[gt.event_label==c]
        ptc=pt.loc[pt.event_label==c]
        
        for f in gtc.filename.append(ptc.filename).unique():
            # if(c=='Blender'):debug=1
            if debug:print(f'============== class={c}=========file={f}')
            g=gtc.loc[gtc.filename==f][['onset','offset']].values
            p=ptc.loc[ptc.filename==f][['onset','offset']].values
            m=(0,10)
            if meta is not None and f in meta: m=meta[f]
            try:
                # debug=c=='Dishes'
                start_time = time.time()
                out=eval_my_metric(g,p,m,debug=debug,calcne=calcne,args=args)
                total_time+=time.time() - start_time
            except Exception as e:
                print(f'============== class={c}=========file={f}')
                print(e)
                import traceback
                print(traceback.format_exc())

                out=eval_my_metric(g,p,m,debug=1,calcne=calcne)

                raise
            if (c not in result):
                result[c]=out
            else:
                for m in out:
                    for t in out[m]:
                        result[c][m][t]+=out[m][t]
    print(f"======= our evaluation method took {total_time} seconds ---- input size: R={len(gt)} P={len(pt)}")
    return result


def intersection(e1,e2):    
    inter=[max(e1[0],e2[0]),min(e1[1],e2[1])]
    if(inter[1]<=inter[0]): 
        inter=None
#     print(e1,e2,inter)
    return inter

def dur(e):
    d= e[1]-e[0]
    if(d<0):
        print(e)
        raise Exception('erorr duration is less than zero')
    return d

def Z(rel,e,X,Y):
    s={}
    for e2 in rel[X][e][Y]:
        for e in rel[Y][e2][X]:
            s[e]=1
    
    return len(s)

def fixLongP(real,pred,rel,ri,pi):
    if(len(rel['p+'][pi]['r+'])>1):
        newp=[pred[pi][0],pred[pi][1]]
        # print(f"p={newp} ri={ri} pi={pi} rel={rel['p+'][pi]['r+']}")
        if (ri-1) in rel['p+'][pi]['r+']:
            newp[0]=(real[ri-1][1]+real[ri][0])/2
            # print(f"ri-1({ri-1}) is in rel==> newp={newp} real[ri-1]={real[ri-1]} real[ri]={real[ri]}")
        if (ri+1) in rel['p+'][pi]['r+']:
            newp[1]=(real[ri][1]+real[ri+1][0])/2
            # print(f"ri+1({ri+1}) is in rel==> newp={newp} real[ri+1]={real[ri+1]} real[ri]={real[ri]}")
        return newp 
    return pred[pi]

def eval_my_metric(real,pred,duration=(0,10),alpha=2,debug=0,calcne=1,args={}):
        prefix='Our: '
        theta_t=args.get('theta_tp', 0)
        theta_f=args.get('theta_fp', 1)
        beta_s=args.get('beta_s',2)
        beta_e=args.get('beta_e',2)
        beta_b=args.get('beta',2)
        pakdd_properties=[
            'detection-pakdd',
            'uniformity-pakdd',
            'total duration',
            'relative duration-pakdd',
            'boundary start-pakdd',
            'boundary end-pakdd',
            # 'boundary both-pakdd'
        ]
        # if debug:debug={'D':1, 'M':1,'U':1, 'T':1, 'R':1,'B':1,'V':1}#V:verbose
        if debug:debug={'D':0, 'M':0,'U':0, 'T':1, 'R':0,'B':0,'V':1, 'DA':1,'UA':1, 'RA':1, 'BSA':1, 'BEA':1,'BBA':1}
        else:debug={'D':0, 'M':0, 'U':0,'T':0, 'R':0,'B':0,'V':0,'DA':0,'UA':0, 'RA':0, 'BSA':0, 'BEA':0,'BBA':0}
        

        # real=merge_events_if_necessary(real)
        # pred=merge_events_if_necessary(pred)
        # real_tree=_makeIntervalTree(real,'r')
        # pred_tree=_makeIntervalTree(pred,'p')
        # duration=(min(duration[0],real[0][0]),max(duration[1],real[-1][1]))
        real=real[real[:,0].argsort(),:]
        pred=pred[pred[:,0].argsort(),:]
        real=np.vstack((real,[duration[1],duration[1]]))# add a zero duration event in the end for ease comparision the last event
        pred=np.vstack((pred,[duration[1],duration[1]]))# add a zero duration event in the end for ease comparision the last event
        #_ means negative
        rel={'r+':{},'r-':{},'p+':{},'p-':{}}
        r_0=(duration[0],real[0][0])
        r_n=(real[-1][1],duration[1])
        metric={}
        pi=0
        rcalc=[]
        real_=[]
        pred_=[]
        ri_=0
        for ri in range(len(real)):
            r=real[ri]
            rp=real[ri-1] if ri>0 else [duration[0],duration[0]]
            r_=(rp[1],r[0])
            
            tmpr={'p+':{},'p-':{}}
            tmpr_={'p+':{},'p-':{}}
            rel['r+'][ri]=tmpr
            
            if(dur(r_)>0):
                real_.append(r_)
                ri_=len(real_)-1
                rel['r-'][ri_]=tmpr_
            
            cond=pi<len(pred) 
            pi_=-1
            # beginProcessed=False
            while  cond:
                pp=pred[pi-1] if pi>0 else [duration[0],duration[0]]
                p=pred[pi]
                p_=(pp[1],p[0])
                if(p_[0]>p_[1]):
                    print(f"pi={pi}{list(enumerate(pred))}")
                if(dur(p_)>0 and (len(pred_)==0 or pred_[-1]!=p_)):
                    pred_.append(p_)
                pi_=len(pred_)-1

                if not(pi in rel['p+']):
                    rel['p+'][pi]={'r+':{},'r-':{}}
                if not(pi_ in rel['p-']):
                    rel['p-'][pi_]={'r+':{},'r-':{}}
                tmpp=rel['p+'][pi]
                tmpp_=rel['p-'][pi_]

                

                rinter  =intersection(r,p)
                rinter_ =intersection(r,p_)
                r_inter =intersection(r_,p)                
                r_inter_=intersection(r_,p_)
                if(rinter is not None):
                    # tmpr['p+'].append((pi,rinter))
                    # tmpp['r+'].append((ri,rinter))
                    tmpr['p+'][pi]=rinter
                    tmpp['r+'][ri]=rinter
                if(rinter_ is not None):
                    # tmpr['p-'].append((pi,rinter_))
                    # tmpp_['r+'].append((ri,rinter_))
                    tmpr['p-'][pi_]=rinter_
                    tmpp_['r+'][ri]=rinter_
                if(r_inter is not None):
                    # tmpr_['p+'].append((pi,r_inter))
                    # tmpp['r-'].append((ri,r_inter))
                    tmpr_['p+'][pi]=r_inter
                    tmpp['r-'][ri_]=r_inter
                if(r_inter_ is not None):
                    # tmpr_['p-'].append((pi,r_inter_))
                    # tmpp_['r-'].append((ri,r_inter_))
                    tmpr_['p-'][pi_]=r_inter_
                    tmpp_['r-'][ri_]=r_inter_
                # if(not beginProcessed):
                #     beginProcessed=true
                #     dur=p[0]-r[0]
                #     if(dur>0):
                #         #underfill start
                #     else:
                #         #overfill end
                        

                
                if pred[pi][1] < r[1]:
                    pi+=1
                else: 
                    cond=False
                    # dur=p[1]-r[1]
                    # if(dur>0):
                    #     #overfill end
                    # else:
                    #     #underfill end

                
            
            # for k in list(rel.keys()):
            #     if len(rel[k])>0: continue
            #     del rel[k]
        
        real=np.delete(real,-1,0)#real.pop()
        pred=np.delete(pred,-1,0)#pred.pop()
#         if(dur(pred_[-1])==0):pred_=np.delete(pred_,-1,0)
#         if(dur(real_[-1])==0):real_=np.delete(real_,-1,0)

        
        out={
            'detection':        {'tp':0,'fp':0,'fn':0,'tn':0},
            'detection-pakdd':   {'tp':0,'fp':0,'fn':0,'tn':0},
            'detect-mono':      {'tp':0,'fp':0,'fn':0,'tn':0},
            'monotony':         {'tp':0,'fp':0,'fn':0,'tn':0},
            'uniformity':       {'tp':0,'fp':0,'fn':0,'tn':0},
            'uniformity-pakdd':       {'tp':0,'fp':0,'fn':0,'tn':0},
            'total duration':   {'tp':0,'fp':0,'fn':0,'tn':0},
            'relative duration':{'tp':0,'fp':0,'fn':0,'tn':0},
            'relative duration-pakdd':{'tp':0,'fp':0,'fn':0,'tn':0},
            'boundary onset':   {'tp':0,'fp':0,'fn':0,'tn':0},
            'boundary offset':  {'tp':0,'fp':0,'fn':0,'tn':0},

            'boundary start-pakdd':   {'tp':0,'fp':0,'fn':0,'tn':0},
            'boundary end-pakdd':  {'tp':0,'fp':0,'fn':0,'tn':0},
            'boundary both-pakdd':  {'tp':0,'fp':0,'fn':0,'tn':0},
            
        }
        
        if debug['V']:
            print("real=",real)
            print("pred=",pred)
            print("real_=",real_)
            print("pred_=",pred_)
            #for x in rel:
            [print(f'{x}: {rel[x]}') for x in rel]
        if debug['V']:plot_events(real,pred,duration,real_,pred_)
        for ri in range(len(real)):
            rdur=dur(real[ri])

            tpd=int(len(rel['r+'][ri]['p+'])>0)
            out['detect-mono']['tp']+=tpd
            out['detection']['tp']+=tpd
            if debug['D']: print(f" D TP+{tpd}      ri={ri}, p+={rel['r+'][ri]['p+']}>0")
            #monotony {
            
            if (len(rel['r+'][ri]['p+'])==1):
                for rpi in rel['r+'][ri]['p+']:
                    if len(rel['p+'][rpi]['r+'])==1:
                        out['monotony']['tp']+=1
                        if debug['M']:print(f"  M TP+1     rel[r+][{ri}][p+]={rel['r+'][ri]['p+']}==1 rel[p+][{rpi}][r+]={rel['p+'][rpi]['r+']}==1")
                    elif(len(rel['p+'][rpi]['r+'])==0):
                        print('error it can not be zero')
                    elif debug['M']:print(f"  M--tp rel[r+][{ri}][p+]={rel['r+'][ri]['p+']}==1 rel[p+][{rpi}][r+]={rel['p+'][rpi]['r+']}>1")
            #}
            pt=0  #for fp detection-pakdd
            ttpr=0 #for tp detection-pakdd
            
            for pi in rel['r+'][ri]['p+']:
                

                tpt=dur(rel['r+'][ri]['p+'][pi])
                tpr=tpt/rdur
                ttpr+=tpr  #for tp detection-pakdd
                
                pt+=dur(fixLongP(real,pred,rel,ri,pi)) #for fp detection-pakdd
                out['total duration']['tp']+=tpt
                out['relative duration']['tp']+=tpr
                if debug['T']:print(f"   T tp+={tpt}             rel[r+][{ri}][p+][{pi}]=dur({rel['r+'][ri]['p+'][pi]})")
                if debug['R']:print(f"    R tp+={tpr}             rel[r+][{ri}][p+][{pi}]==dur({rel['r+'][ri]['p+'][pi]}) / real[{ri}]=dur({real[ri]})")
                
            
            cond_fpr=pt/rdur - ttpr if rdur>0 else 0
            
            cond_fp= int(cond_fpr > theta_f)
            cond_tp=int(ttpr>theta_t)
            out['detection-pakdd']['tp'] += cond_tp
            out['detection-pakdd']['fp'] += cond_fp
            out['relative duration-pakdd']['tp']+=ttpr 
            out['relative duration-pakdd']['fp'] += cond_fpr
            if debug['DA']: print(f" DA TP+{cond_tp} FP+{cond_fp}      ri={ri}, ttpr={ttpr}>theta_t({theta_t}) cond_fpr={cond_fpr}>theta_f({theta_f})")
            if debug['RA']: print(f" RA TP+{ttpr} FP+{cond_fpr}      ri={ri}, ttpr={ttpr} cond_fpr={cond_fpr}")
            ####Uniformity{
            tpuc=Z(rel,ri,'r+','p+')
            
            ###for pakdd
            tpua=int(tpuc==1) 
            out['uniformity-pakdd']['tp']+=tpua
            if debug['UA']:print(f"  UA tp+{tpua}            Z[r+][{ri}][p+]=={tpuc}")
            
            tpu=1/tpuc if tpuc>0 else 0
            if calcne or tpuc>0:
                out['uniformity']['tp']+=tpu
                out['uniformity']['fn']+=1-tpu
                if debug['U']:print(f"  U tp+{tpu}  fn+{1-tpu}           Z[r+][{ri}][p+]=={tpuc}")
            ####Uniformity}

            rps=list(rel['r+'][ri]['p+'].keys())
            ####################boundary onset
            if(len(rps)==0):
                if calcne:
                    out['boundary onset']['fn']+=1
                    out['boundary offset']['fn']+=1
                    if debug['B']:print(f"     B onset&offset fn+{1}  ri={ri} pi=[]          ")
            else:
                # relp=pred[rps[0]]
                relp=fixLongP(real,pred,rel,ri,rps[0])
                boundry_error_s=real[ri][0]-relp[0]
                ufsp=max(0,-boundry_error_s)/rdur
                ofsp=min(1,max(0,boundry_error_s)/rdur)
                tpsp=min(1,max(0,1-ufsp-ofsp))
                out['boundary onset']['tp']+=tpsp
                out['boundary onset']['fn']+=ufsp
                out['boundary onset']['fp']+=ofsp
                if debug['B']:print(f"     B onset tp+{tpsp} fp+{ofsp} fn+{ufsp}  ri={ri} pi={rps[0]}     boundary_error_onset={boundry_error_s}     ")
                
                
                fnsb=1- math.exp(-beta_s * ufsp)
                fpsb=1- math.exp(-beta_s * ofsp)
                tpsb=min(1,max(0,1 - fnsb-fpsb))
                out['boundary start-pakdd']['tp']+=tpsb
                out['boundary start-pakdd']['fn']+=fnsb
                out['boundary start-pakdd']['fp']+=fpsb

                if debug['BSA']:print(f"     BSA onset tp+{tpsb:.2f} fp+{fpsb:.2f} fn+{fnsb:.2f}  ri={ri} pi={rps[0]}     underfill_sr={ufsp:.2f} overfill_sr={ofsp:.2f} dur_s={boundry_error_s:.2f}     ")
                
                #####################boundary offset
                # relp=pred[rps[-1]]
                relp=fixLongP(real,pred,rel,ri,rps[-1])
                boundry_error_e=relp[1]-real[ri][1]
                ufep=min(1,max(0,-boundry_error_e)/rdur)
                ofep=min(1,max(0,boundry_error_e)/rdur)
                tpep=max(0,1-ufep-ofep)
                out['boundary offset']['tp']+=tpep
                out['boundary offset']['fn']+=ufep
                out['boundary offset']['fp']+=ofep
                if debug['B']:print(f"     B offset tp+{tpep} fp+{ofep} fn+{ufep}  ri={ri} pi={rps[-1]}     boundary_error_offset={boundry_error_e}     ")

                fneb=1- math.exp(-beta_e * ufep)
                fpeb=1- math.exp(-beta_e * ofep)
                tpeb=min(1,max(0,1 - fneb-fpeb))
                out['boundary end-pakdd']['tp']+=tpeb
                out['boundary end-pakdd']['fn']+=fneb
                out['boundary end-pakdd']['fp']+=fpeb

                if debug['BEA']:print(f"     BEA onset tp+{tpeb:.2f} fp+{fpeb:.2f} fn+{fneb:.2f}  ri={ri} pi={rps[0]}     underfill_er={ufep:.2f} overfill_er={ofep:.2f} dur_e={boundry_error_e:.2f}     ")

                fnbb=1- math.exp(-beta_b * (ufsp+ufep))
                fpbb=1- math.exp(-beta_b * (ofsp+ofep))
                tpbb=min(1,max(0,1 - fnbb-fpbb))
                out['boundary both-pakdd']['tp']+=tpbb
                out['boundary both-pakdd']['fn']+=fnbb
                out['boundary both-pakdd']['fp']+=fpbb
                if debug['BBA']:print(f"     BBA both tp+{tpbb:.2f} fp+{fpbb:.2f} fn+{fnbb:.2f}  ri={ri} pi={rps[0]}     underfill_br={(ufsp+ufep):.2f} overfill_er={(ofsp+ofep):.2f} dur_s={boundry_error_s:.2f} dur_e={boundry_error_e:.2f}     ")
                

            for pi in rel['r+'][ri]['p-']:
                fnt=dur(rel['r+'][ri]['p-'][pi])
                fnr=fnt/rdur
                out['total duration']['fn']+=fnt
                out['relative duration']['fn']+=fnr if fnr<0.99 else calcne
                if debug['T']:print(f"   T fn+={fnt}             rel[r+][{ri}][p-][{pi}]=dur({rel['r+'][ri]['p-'][pi]})")
                if debug['R']:print(f"    R fn+={fnr}             rel[r+][{ri}][p-][{pi}]==dur({rel['r+'][ri]['p-'][pi]}) / real[{ri}]=dur({real[ri]})")
                

        for ri in range(len(real_)):
            tnd=int(len(rel['r-'][ri]['p-'])>0)
            out['detect-mono']['tn']+=tnd
            out['detection']['tn']+=tnd
            if debug['D']: print(f" D TN+{tnd}      ri-={ri}, p-={rel['r-'][ri]['p-']}>0")
            #monotony {
            
            if (len(rel['r-'][ri]['p-'])==1):
                for rpi in rel['r-'][ri]['p-']:
                    if len(rel['p-'][rpi]['r-'])==1:
                        out['monotony']['tn']+=1
                        if debug['M']:print(f"  M TN+1     rel[r-][{ri}][p-]={rel['r-'][ri]['p-']}==1 rel[p-][{rpi}][r-]={rel['p-'][rpi]['r-']}==1")
                    elif(len(rel['p-'][rpi]['r-'])==0):
                        print('error it can not be zero')
                    elif debug['M']:print(f"  M--tn rel[r-][{ri}][p-]={rel['r-'][ri]['p-']}==1 rel[p-][{rpi}][r-]={rel['p-'][rpi]['r-']}>1")
            #}

            for pi in rel['r-'][ri]['p-']:
                tnt=dur(rel['r-'][ri]['p-'][pi])
                tnr=tnt/dur(real_[ri])
                out['total duration']['tn']+=tnt
                out['relative duration']['tn']+=tnr
                if debug['T']:print(f"   T tn+={tnt}             rel[r-][{ri}][p-][{pi}]=dur({rel['r-'][ri]['p-'][pi]})")
                if debug['R']:print(f"    R tn+={tnr}             rel[r-][{ri}][p-][{pi}]==dur({rel['r-'][ri]['p-'][pi]}) / real_[{ri}]=dur({real_[ri]})")
            for pi in rel['r-'][ri]['p+']:
                fpt=dur(rel['r-'][ri]['p+'][pi])
                fpr=fpt/dur(real_[ri])
                out['total duration']['fp']+=fpt
                out['relative duration']['fp']+=fpr if fpr<0.99 else calcne
                if debug['T']:print(f"   T fp+={fpt}             rel[r-][{ri}][p+][{pi}]=dur({rel['r-'][ri]['p+'][pi]})")
                if debug['R']:print(f"    R fp+={fpr}             rel[r-][{ri}][p+][{pi}]==dur({rel['r-'][ri]['p+'][pi]}) / real_[{ri}]=dur({real_[ri]})")

            if 0:
                rps=list(rel['r-'][ri]['p-'].keys())
                ####################boundary onset
                if(len(rps)==0):
                    out['boundary onset']['fp']+=1
                    out['boundary offset']['fp']+=1
                    if debug['B']:print(f"     B onset&offset fp+{1}  ri-={ri} pi-=[]          ")
                else:
                    relp=pred_[rps[0]]
                    boundry_error_s=real_[ri][0]-relp[0]
                    ufsp=min(1,max(0,-boundry_error_s)/dur(real_[ri]))
                    ofsp=min(1,max(0,boundry_error_s)/dur(real_[ri]))
                    tpsp=max(0,1-ufsp-ofsp)
                    out['boundary onset']['tn']+=tpsp
                    out['boundary onset']['fn']+=ofsp
                    out['boundary onset']['fp']+=ufsp
                    if debug['B']:print(f"     B onset tn+{tpsp} fp+{ufsp} fn+{ofsp}   ri-={ri} pi-={rps[0]}    boundary_error_onset={boundry_error_s}     ")

                    #####################boundary offset
                    relp=pred_[rps[-1]]
                    boundry_error_e=relp[1]-real_[ri][1]
                    ufep=min(1,max(0,-boundry_error_e)/dur(real_[ri]))
                    ofep=min(1,max(0,boundry_error_e)/dur(real_[ri]))
                    tpep=max(0,1-ufep-ofep)
                    out['boundary offset']['tn']+=tpep
                    out['boundary offset']['fn']+=ofep
                    out['boundary offset']['fp']+=ufep
                    if debug['B']:print(f"     B offset tn+{tpep} fp+{ufep} fn+{ofep}   ri-={ri} pi-={rps[-1]}    boundary_error_onset={boundry_error_e}     ")


        out['detect-mono']['fp']=len(real_)-out['detect-mono']['tn']
        if debug['D']: print(f" D fp={out['detect-mono']['fp']} #r-={len(real_)} - tn={out['detect-mono']['tn']}"  )
        out['detect-mono']['fn']=len(real)-out['detect-mono']['tp']
        out['detection']['fn']=len(real)-out['detection']['tp']
        if debug['D']: print(f" D fn={out['detect-mono']['fn']} #r+={len(real)} - tp={out['detect-mono']['tp']}"  )
        out['detection-pakdd']['fn']=len(real)-out['detection-pakdd']['tp']
        if debug['DA']: print(f" DA fn={out['detection-pakdd']['fn']} #r+={len(real)} - tp={out['detection-pakdd']['tp']}"  )
        out['relative duration-pakdd']['fn']=len(real)-out['relative duration-pakdd']['tp']
        if debug['RA']: print(f" RA fn={out['relative duration-pakdd']['fn']} #r+={len(real)} - tp={out['relative duration-pakdd']['tp']}"  )
        out['uniformity-pakdd']['fn']=len(real)-out['uniformity-pakdd']['tp']
        if debug['UA']: print(f" RA fn={out['uniformity-pakdd']['fn']} #r+={len(real)} - tp={out['uniformity-pakdd']['tp']}"  )                
        out['monotony']['fn']=len(real)-out['monotony']['tp']#+len(pred_)-out['monotony']['tn']
        if debug['M']: print(f"  M fn={out['monotony']['fn']}     #r+={len(real)} - tp={out['monotony']['tp']} //+ #p-={len(pred_)} - tn={out['monotony']['tn']}")
        out['monotony']['fp']=len(pred)-out['monotony']['tp']#+len(real_)-out['monotony']['tn']
        if debug['M']: print(f"  M fp={out['monotony']['fp']}     #p+={len(pred)} - tp={out['monotony']['tp']} //+ #r-={len(real_)} - tn={out['monotony']['tn']}")
        
                        
        for pi in range(len(pred)):
            fpd=int(len(rel['p+'][pi]['r+'])==0)
            out['detect-mono']['fp']+=fpd
            out['detection']['fp']+=fpd
            if debug['D']: print(f" D FP+{fpd}      pi={pi}, r={rel['p+'][pi]['r+']}==0")
            out['detection-pakdd']['fp']+=fpd
            if debug['DA']: print(f" DA FP+{fpd}      pi={pi}, r={rel['p+'][pi]['r+']}==0")
            out['relative duration-pakdd']['fp']+=fpd
            if debug['RA']: print(f" RA FP+{fpd}      pi={pi}, r={rel['p+'][pi]['r+']}==0")

            ####Uniformity{
            fpuc=Z(rel,pi,'p+','r+')
            if calcne or fpuc>0:
                fpu=1- (1/fpuc if fpuc>0 else 0)
                out['uniformity']['fp']+=fpu
                if debug['U']:print(f"  U fp+{fpu}             Z[p+][{pi}][r+]=={fpuc}")
            ####Uniformity}
#             for ri in rel['p+'][pi]['r-']:
#                 out['total duration']['fp']+=dur(rel['p+'][pi]['r-'][ri])
#                 out['relative duration']['fp']+=dur(rel['p+'][pi]['r-'][ri])/dur(pred[pi])
        out['uniformity-pakdd']['fp']=len(pred)-out['uniformity-pakdd']['tp']
        if debug['UA']: print(f" RA fp={out['uniformity-pakdd']['fp']} #r+={len(pred)} - tp={out['uniformity-pakdd']['tp']}"  )                

        for pi in range(len(pred_)):
            fnd=int(len(rel['p-'][pi]['r-'])==0)
            out['detect-mono']['fn']+=fnd
            if debug['D']: print(f" D FN+{fnd}      pi-={pi}, r-={rel['p-'][pi]['r-']}==0")
#             for ri in rel['p-'][pi]['r+']:
#                 out['total duration']['fn']+=dur(rel['p-'][pi]['r+'][ri])
#                 out['relative duration']['fn']+=dur(rel['p-'][pi]['r+'][ri])/dur(pred_[pi])

#         plot_events_with_event_scores(range(len(real)),range(len(pred)),real,pred)
#         plot_events_with_event_scores(range(len(real_)),range(len(pred_)),real_,pred_)

        newout={}
        for m in pakdd_properties:
            newout[prefix+m] = out[m]
        # out=dict(filter(lambda elem: elem[0] in pakdd_properties, out.items()))
        if debug['V']:
            [print(m, newout[m]) for m in newout]

        return  newout

def plot_events(real,pred,meta,real_,pred_, label=None):
    from matplotlib.pylab import plt
    import random
    fig,ax = plt.subplots(figsize=(15, .8))
    ax.set_title(label)
    plt.xlim(0,max(meta[1],10))
    ax.set_xticks(np.arange(0,max(real[-1][1],10),.1),minor=True)
    maxsize=20
# random.random()/4
    for i in range(min(maxsize,len(pred_))):
        d = pred_[i]
        plt.axvspan(d[0], d[1], 0, 0.6,linewidth=0,edgecolor='k',facecolor='#edb4b4', alpha=.6)
        plt.text((d[1] + d[0]) / 2, 0.2,f'{i}' , horizontalalignment='center', verticalalignment='center')
    for i in range(min(maxsize,len(pred))):
        d = pred[i]
        plt.axvspan(d[0], d[1], 0.0, 0.6,linewidth=0,edgecolor='k',facecolor='#a31f1f', alpha=.6)
        plt.text((d[1] + d[0]) / 2, 0.2,f'{i}' , horizontalalignment='center', verticalalignment='center')
#     maxsize=len(real)
    for i in range(min(maxsize,len(real_))):
        gt = real_[i]
        plt.axvspan(gt[0], gt[1], 0.4, 1,linewidth=0,edgecolor='k',facecolor='#d2f57a', alpha=.6)
        plt.text((gt[1] + gt[0]) / 2, 0.8,f'{i}' , horizontalalignment='center', verticalalignment='center')
        
    for i in range(min(maxsize,len(real))):
        gt = real[i]
        plt.axvspan(gt[0], gt[1], 0.4, 1,linewidth=0,edgecolor='k',facecolor='#1fa331', alpha=.6)
        plt.text((gt[1] + gt[0]) / 2, 0.8,f'{i}' , horizontalalignment='center', verticalalignment='center')
    # plt.grid(True)
    plt.minorticks_on()
    ax.set(yticks=[.25,.75], yticklabels=['P','R'])
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
