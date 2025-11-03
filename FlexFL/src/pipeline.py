model_name = 'Llama3'
######################### START
from pathlib import Path
PKG_ROOT = Path(__file__).resolve().parents[1]     # .../FlexFL
PROJECT_ROOT = PKG_ROOT.parent                     # .../FlexFL_adapted
DATA_ROOT = PKG_ROOT / "data"
RES_ROOT  = PKG_ROOT / "res"
PREP_ROOT = PROJECT_ROOT / "prepare" / "buggy_program"
BUGGY_INPUT_ROOT = DATA_ROOT / "input" / "buggy_program"

def _buggy_base(dataset: str) -> Path:
    # Prefer data/input/buggy_program/<dataset>, else fallback to prepare/buggy_program/<dataset>
    # or prepare/buggy_program/methods_buggy_<dataset>/
    base = BUGGY_INPUT_ROOT / dataset
    if base.exists():
        return base
    # Try prepare/buggy_program/<dataset>
    fallback = PREP_ROOT / dataset
    if fallback.exists():
        return fallback
    # Try prepare/buggy_program/methods_buggy_<dataset>/ (with case variations)
    fallback2 = PREP_ROOT / f"methods_buggy_{dataset}"
    if fallback2.exists():
        return fallback2
    # Try with lowercase 'j' variant for Defects4J -> Defects4j
    if dataset == "Defects4J":
        fallback3 = PREP_ROOT / "methods_buggy_Defects4j"
        if fallback3.exists():
            return fallback3
    return base

# 1. Build
from typing import List, Optional
from llama import Dialog, Llama
import torch.distributed as dist

ckpt_dir: str = '/home/m.lami/FlexFL_adapted/Meta-Llama-3-8B-Instruct-hf'
tokenizer_path: None
temperature: float = 0
top_p: float = 1.0

seed = 42
max_seq_len: int = 8192
max_batch_size: int = 1
max_gen_len: Optional[int] = None

generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        seed=seed, 
    )

# 2. Inference
def query(instruction):
    results = generator.chat_completion(
                        [instruction],
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
    result = results[0]
    print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
    content = result['generation']['content'].strip()
    return content

######################### END

import json
import os
import shutil
from .function_call import (
    get_code_snippet, get_paths, get_classes, get_methods, find_class, find_method
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Defects4J', choices=['Defects4J', 'GHRB'])
    parser.add_argument('--input', default='All', choices=['bug_report', 'trigger_test', 'All'])
    parser.add_argument('--stage', default='SR', choices=['SR', 'LR'])
    parser.add_argument('--rank', default='All')
    parser.add_argument('--bug-list', default=None, help='Path to custom bug list file (default: data/bug_list/<dataset>/bug_list.txt)')
    args = parser.parse_args()
    dataset = args.dataset
    input_type = args.input
    stage = args.stage
    rank = args.rank

    if args.bug_list:
        bug_list_path = Path(args.bug_list)
    else:
        bug_list_path = DATA_ROOT / "bug_list" / dataset / "bug_list.txt"

    with open(bug_list_path, "r", encoding="utf-8") as f:
        bugs = [e.strip() for e in f.readlines() if e.strip()]

    if stage == "SR":
        output_dir = f"{model_name}_{dataset}_SR" \
                    + ("_br" if input_type == "bug_report" else "") \
                    + ("_tt" if input_type == "trigger_test" else "")
    else:
        output_dir = f"{model_name}_{dataset}_{rank}"

    OUT_DIR = (RES_ROOT / output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for bug in bugs:
        #if bug != 'Time-25':
        #    continue
        print(bug)
        max_try = 10
        while max_try > 0:
            try:
                input_description = ""
                input_type_a = None
                bug_report_path = DATA_ROOT / "input" / "bug_reports" / dataset / f"{bug}.json"
                if input_type in ("All", "bug_report") and bug_report_path.exists():
                    input_type_a = "a bug report"
                    with bug_report_path.open("r", encoding="utf-8") as f:
                        bug_report = json.load(f)
                    title = (bug_report.get("title") or "").replace("\n", " ").strip()
                    desc  = (bug_report.get("description") or bug_report.get("description_text") or "").replace("\n", " ").strip()
                    input_description += (
                        "The bug report is as follows:\n```\n"
                        f"Title:{title}\nDescription:{desc}\n```\n"
                    )

                trigger_test_path = DATA_ROOT / "input" / "trigger_tests" / dataset / f"{bug}.txt"
                if input_type in ("All", "trigger_test") and trigger_test_path.exists():
                    input_type_a = ("a bug report, a trigger test" if input_type_a else "a trigger test")
                    trigger_test = trigger_test_path.read_text(encoding="utf-8")
                    input_description += f"The trigger test is as follows:\n```\n{trigger_test}\n```\n"
                #input_type_the = (input_type_a or "").replace('a', 'the') if input_type_a else "the input"
                input_type_the = (input_type_a or "the input")
                if stage == 'SR':
                    functions = "\nFunction calls you can use are as follows.\n\
* find_class(`class_name`) -> Find a class in the bug report by fuzzy search. `class_name` -> The name of the class. *\n\
* find_method(`method_name`) -> Find a method in the bug report by fuzzy search. `method_name` -> The name of the method. *\n\
* get_paths() -> Get the paths of the java software system. *\n\
* get_classes_of_path(`path_name`) -> Get the classes in the path of the java software system. `path_name` -> The accurate name of the path which can be accessed by function call `get_paths`. *\n\
* get_methods_of_class(`class_name`) -> Get the methods belonging to the class of the java software system. `class_name` -> The complete name of the class, for example `PathName.ClassName`. *\n\
* get_code_snippet_of_method(`method_name`) -> Get the code snippet of the java method. `method_name` -> The complete name of the java method, for example `PathName.ClassName.MethodName(ArgType1,ArgType2)`. *\n\
* exit() -> Exit function calling to give your final answer when you are confident of the answer. *\n"
                else:
                    functions = "\nFunction calls you can use are as follows.\n\
* get_code_snippet_of_method(`method_number`) -> Get the code snippet of the java method. `method_number` -> The number of the java methods suggested. *\n\
* exit() -> Exit function calling to give your final answer when you are confident of the answer.  *\n"
                    sus_paths = DATA_ROOT / "input" / "suspicious_methods" / dataset / f"{model_name}_{rank}" / f"{bug}.txt"
                    with sus_paths.open("r", encoding="utf-8") as f:
                        suspicious_methods = f.read().splitlines()
                        suspicious_methods_content =  '\n'.join([f"{j}.{suspicious_methods[j-1]}" for j in range(1,len(suspicious_methods)+1)])
                        
                    input_description += f"The suggested methods are as follows:\n```\n{suspicious_methods_content}\n```\n"
                instruction = [
                    {
                        "role": "system",
                        "content": f"You are a debugging assistant of our Java software. \
You will be presented with {input_type_a} and tools (functions) to access the source code of the system under test (SUT). \
Your task is to locate the top-5 most likely culprit methods based on {input_type_the} and the information you retrieve using given functions. {functions}\
You have {max_try} chances to call function."
    },
                    {
                        "role": "user",
                        "content": f"{input_description}\
Let's locate the faulty method step by step using reasoning and function calls. \
Now reason and plan how to locate the buggy methods."
    }
                ]
                content = query(instruction)
                instruction.append({
                    "role": "Assistant",
                    "content": content
                })
                for j in range(max_try):
                    instruction.append({
                        "role": "user",
                        "content": f"Now call a function in this format `FunctionName(Argument)` in a single line without any other word."
                    })
                    content = query(instruction)
                    instruction.append({
                        "role": "Assistant",
                        "content": content
                    })
                    try:
                        function_call = content.replace("'","").replace('"','')
                        function_name = function_call[:function_call.find('(')].strip()
                        arguments = function_call[function_call.find('(')+1:function_call.rfind(')')].strip().strip('`')
                        if function_name == 'get_paths':
                            function_retval = get_paths(bug, dataset)
                        elif function_name == 'get_classes_of_path':
                            function_retval = get_classes(bug, arguments, dataset)
                        elif function_name == 'get_methods_of_class':
                            function_retval = get_methods(bug, arguments, dataset)
                        elif function_name == 'get_code_snippet_of_method':
                            if stage == 'SR':
                                function_retval = get_code_snippet(bug, arguments, dataset)
                            else:
                                function_retval = get_code_snippet(bug, suspicious_methods[int(arguments)-1], dataset)
                                function_retval = f"The code snippet of {suspicious_methods[int(arguments)-1]} is as follows.\n" + function_retval
                        elif function_name == 'find_class':
                            function_retval = find_class(bug, arguments, dataset)
                        elif function_name == 'find_method':
                            function_retval = find_method(bug, arguments, dataset)
                        elif function_name == 'exit':
                            break
                        else:
                            instruction.append({
                            "role": "user",
                            "content": "Please call functions in the right format `FunctionName(Argument)`." + functions})
                            continue
                        print(function_retval)
                        instruction.append({"role": "user", "content": function_retval})
                    except Exception as e:
                        print(e)
                        instruction.append({
                            "role": "user",
                            "content": "Please call functions in the right format `FunctionName(Argument)`." + functions})
                instruction.append({
            "role": "user",
            "content": "Based on the available information, provide complete name of the \
top-5 most likely culprit methods for the bug please. \
Since your answer will be processed automatically, please give your answer in the format as follows.\n\
Top_1 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n\
Top_2 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n\
Top_3 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n\
Top_4 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n\
Top_5 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n\
"
        })
                content = query(instruction)
                instruction.append({
                        "role": "Assistant",
                        "content": content
                    })
                # final write
                (out_path := OUT_DIR / f"{bug}.json").write_text(json.dumps(instruction, indent=4), encoding="utf-8")
                break
            except Exception as e:
                print(e)
                max_try -= 1
