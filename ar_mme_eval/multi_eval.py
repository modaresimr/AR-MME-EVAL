import numpy as np
import pandas as pd
import os
import glob

from ar_mme_eval.other_eval import psds_score, compute_psds_from_operating_points, compute_metrics
from . import other_eval
import psds_eval
import warnings


def psds_metric(dtc_threshold, gtc_threshold, cttc_threshold, ground_truth, metadata, predictions):
    psds = psds_eval.PSDSEval(dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold, cttc_threshold=cttc_threshold, ground_truth=ground_truth, metadata=metadata.reset_index().rename(columns={'index': 'filename'}))
    # from psds_macro_f1, psds_f1_classes = psds.compute_macro_f_score(predictions)
    det_t = psds._init_det_table(predictions)
    counts, tp_ratios, _, _ = psds._evaluate_detections(det_t)
    per_class_tp = np.diag(counts)[:-1]

    num_gts = np.divide(per_class_tp, tp_ratios, out=np.zeros_like(per_class_tp), where=tp_ratios != 0)

    per_class_fp = counts[:-1, -1]
    per_class_fn = num_gts - per_class_tp
    classes = sorted(set(psds.class_names).difference([psds_eval.psds.WORLD]))

    dic = {c: {'Ntp': tp, 'Nfp': fp, 'Nfn': fn} for c, tp, fp, fn in zip(classes, per_class_tp, per_class_fp, per_class_fn)}

    return dic


def compute_mme(ground_truth, metadata, predictions, debug=0, args={}):
    from . import mme as m
    ev = m.eval(ground_truth, predictions, metadata, debug=debug, args=args)
    mm = ev[list(ev.keys())[0]].keys()
    # out={m:{c:{'Ntp':ev[c][m]['tp'],'Nfp':ev[c][m]['fp'],'Nfn':ev[c][m]['fn'],'Ntn':ev[c][m]['tn']}  for c in ev} for m in mm}
    out = {m: {c: {'Ntp': ev[c][m]['tp'], 'Nfp': ev[c][m]['fp'], 'Nfn': ev[c][m]['fn']} for c in ev} for m in mm}
    return out


def get_single_result(gtf, pef, metaf=None, psdsf=None, debug=0, args={}):
    res = {'macro_avg', 'micro_avg', 'class'}

    # gem=computeGem(gtf,pef)

    groundtruth = pd.read_csv(gtf, comment='#', sep="\t")
    # Evaluate a single prediction
    predictions = pd.read_csv(pef, comment='#', sep="\t")
    print("".join([line for line in open(pef)if line.startswith("#")]))
    meta_df = None
    if(metaf is not None):
        meta_df = pd.read_csv(metaf, comment='#', sep="\t")

    # print(meta_df)
    return get_single_result_df(groundtruth, predictions, meta_df, debug=debug, args=args)


def get_single_result_df(groundtruth, predictions, meta_df=None, psdsf=None, debug=0, args={}):
    out = {}
    if meta_df is None:
        meta_df = pd.DataFrame(groundtruth.append(predictions).groupby(['filename'])['offset'].max().rename('duration'))
        meta_df[meta_df['duration'] < 10] = 10

    if 'filename' in meta_df.columns:
        meta_df = meta_df.set_index('filename')

    def calcs(metric,args={}):
        df = pd.DataFrame(metric).T
        df.loc['micro-avg'] = df.sum()
        df['recall'] = df['Ntp']/(df['Ntp']+df['Nfn'])
        df['precision'] = df['Ntp']/(df['Ntp']+df['Nfp'])
        df['f1'] = 2*df['precision']*df['recall']/(df['precision']+df['recall'])
        df.loc['macro-avg'] = df.drop('micro-avg').mean()
        # print(df)
        # df['f1']=2*df['precision']*df['recall']/(df['precision']+df['recall'])
        # return df[['Ntp','Nfp','Nfn','recall','precision','f1']]
        if(args.get('show tp/fp/fn',False)):
            return df[['Ntp', 'Nfp', 'Nfn', 'recall', 'precision', 'f1']]
        return df[['recall', 'precision', 'f1']]
    if args.get('show_sed', 0):
        events_metric = other_eval.event_based_evaluation_df(groundtruth, predictions, t_collar=0.200, percentage_of_length=0.2)
        events_metric_df = calcs(events_metric.class_wise)
        out["collar"] = events_metric_df
        # print('events_metric',events_metric)
        # groundtruth=groundtruth[groundtruth['event_label']=='Blender']
        # predictions=predictions[predictions['event_label']=='Blender']
        segment_metric = other_eval.segment_based_evaluation_df(groundtruth, predictions, meta_df, time_resolution=1.)
        # print(segment_metric.class_wise)
        # print('segment_metric',segment_metric)
        segment_metric_df = calcs(segment_metric.class_wise)

        out["segment"] = segment_metric_df
        #macro_f1_event = events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
        #macro_f1_segment = segment_metric.results_class_wise_average_metrics()['f_measure']['f_measure']

        thresh = np.arange(0.1, 1, .2)  # np.arange(0.1,1,.6)=[0.1,0.7]  ,np.arange(0.1,1,.2)=[0.1, 0.3, 0.5, 0.7, 0.9]
        thresh = [0.1, 0.3, 0.5, 0.8, 0.85, 0.9]
        for t in thresh:
            psds = psds_metric(dtc_threshold=t, gtc_threshold=t, cttc_threshold=.3, ground_truth=groundtruth, metadata=meta_df, predictions=predictions)
            psds_df = calcs(psds)
            out[f'psd d/gtc={t}'] = psds_df.round(2)

    metadata = {}
    for i, f in meta_df.iterrows():
        metadata[i] = (0, f['duration'])

    mme = compute_mme(groundtruth, metadata, predictions, debug=debug, args=args)
    for m in mme:
        out[m] = calcs(pd.DataFrame(mme[m]),args).round(2)

    ward = ward_metric(groundtruth, predictions, args=args)
    for m in ward:
        out[m] = ward[m].round(2)
    tat = tatbul_metric(groundtruth, predictions, args=args)
    for m in tat:
        out[m] = tat[m].round(2)
    # print(out)
    return out


def ward_metric(groundtruth, predictions, args={}):
    from . import ward as ward
    res = ward.eval(groundtruth, predictions, args=args)
    # from psds_macro_f1, psds_f1_classes = psds.compute_macro_f_score(predictions)
    # print(tat)
    clazz = list(res.keys())
    metrics = list(res[clazz[0]].keys())
    # print(res)
    out = {m: pd.DataFrame({c: {'rate': res[c][m]} for c in clazz}).T for m in metrics}
    for m in out:
        out[m].loc['macro-avg'] = out[m].mean()
        # out[m].loc['micro-avg'] = 0

    return out


def tatbul_metric(groundtruth, predictions, args={}):
    from . import tatbul as tatbul

    tat = tatbul.eval(groundtruth, predictions, args=args)
    # from psds_macro_f1, psds_f1_classes = psds.compute_macro_f_score(predictions)
    # print(tat)
    clazz = list(tat.keys())
    metrics = list(tat[clazz[0]].keys())
    out = {m: pd.DataFrame({c: {'recall': tat[c][m]['recall'], 'precision': tat[c][m]['precision'], 'f1': tat[c][m]['f1']} for c in clazz}).T for m in metrics}
    for m in out:
        out[m].loc['macro-avg'] = out[m].mean()
        out[m].loc['micro-avg'] = 0

    return out
