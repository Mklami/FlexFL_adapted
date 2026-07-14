import json
import os
import argparse
import csv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_class_from_method(method_name):
    """
    Extract class name from a fully qualified method name.
    Example: 'org.jfree.chart.plot.CategoryPlot.getDataset(int)' -> 'org.jfree.chart.plot.CategoryPlot'
    """
    # Remove method arguments if present
    if '(' in method_name:
        method_name = method_name[:method_name.rindex('(')]
    # Get everything before the last dot (which separates class from method)
    if '.' in method_name:
        parts = method_name.rsplit('.', 1)
        return parts[0]  # Return the class name
    return method_name  # Fallback if no dot found


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Defects4J', choices=['Defects4J'])
    parser.add_argument('--bug_list', default='All', choices=['All','AutoFL'])
    parser.add_argument('--model', default='Llama3')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    results_dir = f'./{model}_{dataset}_All'

    if args.bug_list == 'All':
        bug_file = os.path.join(BASE_DIR, "bug_list.txt")
    elif args.bug_list == 'AutoFL':
        bug_file = os.path.join(BASE_DIR, "bug_list_AutoFL.txt")
        with open(f'./bug_list_AutoFL.txt') as f:
            bugs = [e.strip() for e in f.readlines()]

    with open(bug_file) as f:
            bugs = [e.strip() for e in f.readlines()]


    
    gt_path = os.path.join(
        BASE_DIR,
        "..", "data", "input", "ground_truth", dataset, "gt.json"
    )
    with open(gt_path) as f:
            gt = json.load(f)

    top1_cnt = 0
    top1 = []
    top3_cnt = 0
    top3 = []
    top5_cnt = 0
    top5 = []
    bug_cnt = 0

    MAP = 0
    MRR = 0
    
    # Store per-bug metrics for CSV
    per_bug_results = []

    for bug in bugs:
        json_path = os.path.join(results_dir, f'{bug}.json')
        if not os.path.exists(json_path):
            print(f"Skipping {bug} - JSON file not found")
            continue
        
        with open(json_path) as f:
            res = json.load(f)

        bug_cnt += 1
        res = res[-1]['content']
        suspicious_methods = []
        for line in res.split('\n'):
            for i in range(1,6):
                if f'Top_{i} : ' in line:
                    suspicious_methods.append(line.split(f'Top_{i} : ')[1].strip())
                elif f'Top_{i}: ' in line:
                    suspicious_methods.append(line.split(f'Top_{i}: ')[1].strip())
                elif f'Top {i}: ' in line:
                    suspicious_methods.append(line.split(f'Top {i}: ')[1].strip())
        
        # Extract files from ground truth methods
        gt_files = set()
        for gt_method in gt[bug]:
            class_name = extract_class_from_method(gt_method)
            gt_files.add(class_name)
        
        # Extract files from suspicious methods
        suspicious_files = []
        for method in suspicious_methods:
            class_name = extract_class_from_method(method)
            suspicious_files.append(class_name)
        
        # Calculate per-bug metrics
        top1_hit = 0
        top3_hit = 0
        top5_hit = 0
        top1_file_hit = 0
        top3_file_hit = 0
        top5_file_hit = 0
        bug_mrr = 0.0
        bug_map = 0.0
        
        # Check top-1, top-3, top-5 hits (method-level)
        for i in [1,3,5]:
            flag = False
            for method in suspicious_methods[:i]:
                if method in gt[bug]:
                    flag = True
                    break

            if flag:
                if i == 1: 
                    top1_cnt += 1
                    top1.append(bug)
                    top1_hit = 1
                if i == 3: 
                    top3_cnt += 1
                    top3.append(bug)
                    top3_hit = 1
                if i == 5: 
                    top5_cnt += 1
                    top5.append(bug)
                    top5_hit = 1
        
        # Find the rank of first file hit
        file_hit_rank = 0
        for rank in range(1, min(6, len(suspicious_files) + 1)):  # Check ranks 1-5
            if suspicious_files[rank - 1] in gt_files:  # rank-1 because list is 0-indexed
                file_hit_rank = rank
                break
        
        # Check top-1, top-3, top-5 file-level hits
        for i in [1,3,5]:
            file_flag = False
            top_i_files = set(suspicious_files[:i])
            if top_i_files & gt_files:  # Check if there's any intersection
                file_flag = True
                if i == 1:
                    top1_file_hit = 1
                if i == 3:
                    top3_file_hit = 1
                if i == 5:
                    top5_file_hit = 1
        
        # Calculate MRR for this bug
        for i in range(min(5,len(suspicious_methods))):
            method = suspicious_methods[i]
            if method in gt[bug]:
                bug_mrr = 1 / (i+1)
                MRR += bug_mrr
                break
        
        # Calculate MAP for this bug
        avg_pre = 0
        for gt_method in gt[bug]:
            num = 0
            for i in range(min(5,len(suspicious_methods))):
                method = suspicious_methods[i]
                if method in gt[bug]:
                    num += 1
                if method == gt_method:
                    precision = num / (i+1)
                    avg_pre += precision
                    break
        if len(gt[bug]) != 0:
            avg_pre /= len(gt[bug])
        bug_map = avg_pre
        MAP += bug_map
        
        # Store per-bug results
        per_bug_results.append({
            'bug_id': bug,
            'top1_hit': top1_hit,
            'top3_hit': top3_hit,
            'top5_hit': top5_hit,
            'top1_file_hit': top1_file_hit,
            'top3_file_hit': top3_file_hit,
            'top5_file_hit': top5_file_hit,
            'file_hit_rank': file_hit_rank,  # Rank where first file hit occurs (0 if none)
            'mrr': bug_mrr,
            'map': bug_map,
            'num_gt_methods': len(gt[bug]),
            'num_gt_files': len(gt_files)
        })
    if args.bug_list == 'All':
        print('Results of FlexFL in Table4:')
    else:
        print('Results of FlexFL in Table5:')
    print("Top-1", top1_cnt)
    print("Top-3", top3_cnt)
    print("Top-5", top5_cnt)
    print("All", bug_cnt)
    print("MAP", MAP / bug_cnt)
    print("MRR", MRR / bug_cnt)
    res = {
        "Total": bug_cnt,
        "Top-1": top1_cnt,
        "Top-3": top3_cnt,
        "Top-5": top5_cnt,
        "MAP": MAP / bug_cnt,
        "MRR": MRR / bug_cnt,
        "top-1": top1,
        "top-3": top3,
        "top-5": top5
    }
    if args.bug_list == 'All':
        with open(f'./res_Table_4.json', 'w') as f:
            json.dump(res,f,indent=4)
        csv_filename = './per_bug_metrics_Table_4.csv'
    else:
        with open(f'./res_Table_5.json', 'w') as f:
            json.dump(res,f,indent=4)
        csv_filename = './per_bug_metrics_Table_5.csv'
    
    # Write per-bug metrics to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['bug_id', 'top1_hit', 'top3_hit', 'top5_hit', 'top1_file_hit', 'top3_file_hit', 'top5_file_hit', 'file_hit_rank', 'mrr', 'map', 'num_gt_methods', 'num_gt_files']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_bug_results)
    
    print(f"\nPer-bug metrics saved to {csv_filename}")