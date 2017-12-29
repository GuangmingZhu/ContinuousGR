import os
import sys
# import data_io
import tools
import jaccard_index as ji
import traceback
DEBUG = True

def main():

    try:
        # get the ref file
        ref_path = sys.argv[1]
        # get the res file
        res_path = sys.argv[2]
        # build ref dict
        gt_table, js_table = tools.read_formatted_file(ref_path)
        p_table, _ = tools.read_formatted_file(res_path)
        for video in p_table:
            ps = p_table[video]
            gts = gt_table[video]
            labels = set()
            g_labels = set()
            for seg in gts:
                _, l = seg
                labels.add(l)
                g_labels.add(l)
            for seg in ps:
                _, l = seg
                labels.add(l)
            sum_jsi_v = 0.
            for label in labels:
                jsi_value = ji.Jsi(gts, ps, label)
                sum_jsi_v += jsi_value
            Js = sum_jsi_v / len(g_labels)
            js_table[video] = Js
        mean_jaccard_index = sum(js_table.values()) / float(len(js_table))
    except Exception as err:
        print err
        print traceback.print_exc()
        return
    score_result = open(sys.argv[3], 'wb')
    score_result.write("Accuracy: %0.6f\n" % mean_jaccard_index)
    score_result.close()
    print mean_jaccard_index
if __name__ == '__main__':
    main()
    # if DEBUG:
    #     data_io.show_io(input_dir, output_dir)
