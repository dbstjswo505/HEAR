#!/bin/bash

. path.sh

if [ $# -lt 1 ]; then
    echo "DSTC7-AVSD objective evaluation"
    echo "Usage: dstc7avsd_eval.sh result1.json [result2.json ..]"
    echo " output: result1.eval [result2.eval ..]"
    exit
fi

# arguments
myArray=("$@")
# references
reference=data/test_set4DSTC7-AVSD_multiref.json

# ms-coco setup
if [ ! -d utils/coco-caption ]; then
    echo setup ms-coco evaluation tool
    git clone https://github.com/tylin/coco-caption utils/coco-caption
    patch -p0 -u < utils/coco-caption.patch
fi
# do evaluation
for result in "${myArray[@]}"; do
    echo "Result: $result"
    hypothesis="${result%.*}_hyp.json"
    result_eval="${result%.*}.eval"
    python utils/get_hypotheses.py -l -s data/stopwords.txt "$result" "$hypothesis"
    python utils/evaluate.py "$reference" "$hypothesis" >& "$result_eval"
    echo "--- summary ---"
    awk '/^(Bleu_[1-4]|METEOR|ROUGE_L|CIDEr):/{print $0; if($1=="CIDEr:"){exit}}'\
        "$result_eval"
    echo "---------------"
    rm "$hypothesis"
done
