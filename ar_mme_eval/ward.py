
from wardmetrics.core_methods import eval_events, eval_segments
from wardmetrics.utils import *
from wardmetrics.visualisations import *
import numpy as np


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
                out = eval_ward_metric(g, p, args=args)
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
    print(f"======= ward evaluation method took {total_time} seconds ---- input size: R={len(gt)} P={len(pt)}")
    return result


def eval_ward_metric(real, pred, args={}):
    if(len(real) == 0 or len(pred) == 0):
        print("No Prediction or Ground Truth -> Ward metric can not be ploted")
        return {
            'Ward Time: true positive rate':			0,
            'Ward Time: deletion rate':				0,
            'Ward Time: fragmenting rate':			0,
            'Ward Time: start underfill rate':		0,
            'Ward Time: end underfill rate':			0,

            'Ward Time: 1-false positive rate':	0,
            'Ward Time: insertion rate':				0,
            'Ward Time: merge rate':					0,
            'Ward Time: start overfill rate':		0,
            'Ward Time: end overfill rate':			0,
            'Ward Event: deletions/gt': 0,
            'Ward Event: merged/gt': 0,
            'Ward Event: fragmented/gt': 0,
            'Ward Event: frag. and merged/gt': 0,
            'Ward Event: correct/gt': 0,


            'Ward Event: insertions/pr': 0,
            'Ward Event: merging/pr':  0,
            'Ward Event: fragmenting/pr': 0,
            'Ward Event: frag. and merging/pr': 0,
            'Ward Event: correct/pr': 0,
        }

    # m =  min(min(pred[:, 0]),min(real[:, 0]))

    ground_truth_test = list(zip(real[:, 0], real[:, 1]))
    detection_test = list(zip(pred[:, 0], pred[:, 1]))

    twoset_results, segments_with_scores, segment_counts, normed_segment_counts = eval_segments(ground_truth_test, detection_test)

    fn = segment_counts["D"] + segment_counts["F"] + segment_counts["Us"] + segment_counts["Ue"]
    fp = segment_counts["I"] + segment_counts["M"] + segment_counts["Os"] + segment_counts["Oe"]

    twoset = {
        'Ward Time: true positive rate':			twoset_results["tpr"],
        'Ward Time: deletion rate':				twoset_results["dr"],
        'Ward Time: fragmenting rate':			twoset_results["fr"],
        'Ward Time: start underfill rate':		twoset_results["us"],
        'Ward Time: end underfill rate':			twoset_results["ue"],
        'Ward Time: 1-false positive rate':	    1-twoset_results["fpr"],
        'Ward Time: insertion rate':				twoset_results["ir"],
        'Ward Time: merge rate':					twoset_results["mr"],
        'Ward Time: start overfill rate':		twoset_results["os"],
        'Ward Time: end overfill rate':			twoset_results["oe"]
    }

    recall = segment_counts['TP']/(segment_counts['TP']+fn)
    prec = segment_counts['TP']/(segment_counts['TP']+fp)
    f1 = 2*recall*prec/(recall+prec+0.000001)
    # print({'recall': recall, 'precision': prec, 'f1': f1})
    # Print results:

    # Run Ward Event: based evaluation:
    gt_event_scores, det_event_scores, detailed_scores, standard_scores = eval_events(ground_truth_test, detection_test)

    evnt = {

        'Ward Event: deletions/gt': detailed_scores["D"]/detailed_scores["total_gt"],
        'Ward Event: merged/gt': detailed_scores["M"]/detailed_scores["total_gt"],
        'Ward Event: fragmented/gt': detailed_scores["F"]/detailed_scores["total_gt"],
        'Ward Event: frag. and merged/gt': detailed_scores["FM"]/detailed_scores["total_gt"],
        'Ward Event: correct/gt': detailed_scores["C"]/detailed_scores["total_gt"],


        'Ward Event: insertions/pr': detailed_scores["I'"]/detailed_scores["total_det"],
        'Ward Event: merging/pr':  detailed_scores["M'"]/detailed_scores["total_det"],
        'Ward Event: fragmenting/pr': detailed_scores["F'"]/detailed_scores["total_det"],
        'Ward Event: frag. and merging/pr': detailed_scores["FM'"]/detailed_scores["total_det"],
        'Ward Event: correct/pr': detailed_scores["C"]/detailed_scores["total_det"],
        #print("\trecall:\t\t" + str(standard_event_results["recall"]) + "\tWeighted by length:\t" + str(standard_event_results["recall (weighted)"]))

    }
    # Print results:
    # print_standard_event_metrics(standard_scores)

    # Visualisations:
    if(args.get('plot_ward', 0)):
        plot_segment_counts(segment_counts)
        plot_twoset_metrics(twoset_results, startangle=45)
        # Show results:
        plot_events_with_event_scores(gt_event_scores, det_event_scores, ground_truth_test, detection_test, show=False)
        plot_event_analysis_diagram(detailed_scores)
        print_detailed_segment_results(normed_segment_counts)
        print_twoset_segment_metrics(twoset_results)
        print_detailed_event_metrics(detailed_scores)

    return {**evnt, **twoset}
