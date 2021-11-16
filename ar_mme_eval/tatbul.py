
def eval(gt, pt, meta=None, clas=None, args={}):
    # %matplotlib inline
    if clas is None:
        clas = gt.event_label.append(pt.event_label).unique()

    result = {}
    total_time = 0
    import time

    for c in clas:
        gtc = gt.loc[gt.event_label == c]
        ptc = pt.loc[pt.event_label == c]

        files = gtc.filename.append(ptc.filename).unique()
        if(len(files) > 1):
            print("Warning only first file is evaluated using tatbul")
        for f in gtc.filename.append(ptc.filename).unique():
            # if(c=='Blender'):debug=1
            if args.get('debug', 0):
                print(f'============== class={c}=========file={f}')
            g = gtc.loc[gtc.filename == f][['onset', 'offset']].values
            p = ptc.loc[ptc.filename == f][['onset', 'offset']].values
            m = (0, 10)
            if meta is not None and f in meta:
                m = meta[f]
            try:
                # debug=c=='Dishes'
                start_time = time.time()
                out = eval_tatbul_metric(g, p, args=args)
                total_time += time.time() - start_time
            except Exception as e:
                print(f'============== class={c}=========file={f}')
                print(e)
                import traceback
                print(traceback.format_exc())

                # out = eval_tatbul_metric(g, p, args=args)

                raise
            if (c not in result):
                result[c] = out
            else:
                for m in out:
                    for t in out[m]:
                        result[c][m][t] += out[m][t]
            break
    print(f"======= tatbul evaluation method took {total_time} seconds ---- input size: R={len(gt)} P={len(pt)}")
    return result


def eval_tatbul_metric(real, pred, args={}):
    m = 0
    if(len(pred) != 0):
        m = min(pred[:, 0])
    if(len(real) != 0):
        m = min(m, min(real[:, 0]))

    reventf = open(f"/tmp/tat.realn", "w")
    peventf = open(f"/tmp/tat.predn", "w")

    for ri in range(len(real)):
        r = real[ri]
        reventf.write(f'{r[0]-m} {r[1]-m}\n')

    for pi in range(len(pred)):
        p = pred[pi]
        peventf.write(f'{p[0]-m} {p[1]-m}\n')

    reventf.close()
    peventf.close()

    result = call_tatbul_metric_all(f"/tmp/tat.realn", f"/tmp/tat.predn", args=args)
    return result


def call_tatbul_metric_all(real, pred, args={}):
    result = {}
    result['Tatbul: a=0, γ=1, δ=back'] = call_tatbul_metric(real, pred, flag='-t', beta=1, alpha=0, gamma='one', delta='back')
    result['Tatbul: a=0, γ=1, δ=middle'] = call_tatbul_metric(real, pred, flag='-t', beta=1, alpha=0, gamma='one', delta='middle')
    result['Tatbul: a=0, γ=1, δ=front'] = call_tatbul_metric(real, pred, flag='-t', beta=1, alpha=0, gamma='one', delta='front')
    result['Tatbul: a=0, γ=1, δ=flat'] = call_tatbul_metric(real, pred, flag='-t', beta=1, alpha=0, gamma='one', delta='flat')
    result['Tatbul: a=1, γ=1, δ=flat'] = call_tatbul_metric(real, pred, flag='-t', beta=1, alpha=1, gamma='one', delta='flat')
    result['Tatbul: a=0, γ=reci, δ=flat'] = call_tatbul_metric(real, pred, flag='-t', beta=1, alpha=0, gamma='reciprocal', delta='flat')
    # result['Tatbul Classical'] = call_tatbul_metric(real, pred, flag='-c', beta=1, alpha=1, gamma='one', delta='flat')
    # result['Tatbul Numenta'] = call_tatbul_metric(real, pred, flag='-n', beta=1, alpha=1, gamma='one', delta='flat')

    return result
    # print(a)
    # # os.system(f'/workspace/TSAD-Evaluator/src/evaluate -v -tn {real} {pred} 1 0 one flat flat')
    # print(f'/workspace/TSAD-Evaluator/src/evaluate -v -tn {real} {pred} 1 0 one flat flat');


def call_tatbul_metric(real, pred, flag, beta, alpha, gamma, delta):
    import os
    import subprocess
    # print(f'/workspace/TSAD-Evaluator/src/evaluate -v -tn {real} {pred} {str(beta)} {str(alpha)} {gamma} {delta} {delta}')
    proc = subprocess.run(['/workspace/TSAD-Evaluator/src/evaluate',  '-i', flag, f'{real}', f'{pred}', str(beta), str(alpha), gamma, delta, delta],  stdout=subprocess.PIPE)
    a = proc.stdout.decode('utf-8')
    # print(a)
    terms = {'Precision': 'precision', 'Recall': 'recall', 'F-Score': 'f1'}
    result = {}
    for x in a.split('\n'):
        for term in terms:
            if(term in x):
                result[terms[term]] = float(x[len(term)+3:])

    return result


if __name__ == "__main__":
    pass
