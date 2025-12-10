CURRENT_PATH=$(readlink -f "$0")
CURRENT_DIR=$(dirname "$CURRENT_PATH")

TOOL_DIR=$(cd $CURRENT_DIR; cd ..; cd ..; pwd)/tools
export D4J_HOME=$TOOL_DIR/defects4j
REPO_DIR=$CURRENT_DIR/Collect_Methods/repos

cat "./bug_list.txt" | while read bug
do
    project=${bug%-*}
    bugid=${bug#*-}
    checkout_dir=$REPO_DIR/${project}-${bugid}
    echo $project-$bugid
    
    # Skip if buggy version already exists
    if [ -d "${checkout_dir}_buggy" ]; then
        echo "  Skipping buggy checkout (already exists)"
    else
        rm -rf "${checkout_dir}_buggy"; mkdir -p "${checkout_dir}_buggy"
        "$D4J_HOME/framework/bin/defects4j" checkout -p "$project" -v "${bugid}b" -w "${checkout_dir}_buggy"
    fi
    
    # Skip if fixed version already exists
    if [ -d "${checkout_dir}_fixed" ]; then
        echo "  Skipping fixed checkout (already exists)"
    else
        rm -rf "${checkout_dir}_fixed"; mkdir -p "${checkout_dir}_fixed"
        "$D4J_HOME/framework/bin/defects4j" checkout -p "$project" -v "${bugid}f" -w "${checkout_dir}_fixed"
    fi
done