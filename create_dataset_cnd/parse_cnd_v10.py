"""
This file will generate citation and abtract info from Citation Network Dataset v10
Input: citation network dataset v10 from "https://www.aminer.org/citation"
Output: 1. dataset contains all interested conferences since 2000 (format: id, conf, year, author, title, n_citation, reference, abstract)
        2. number of paper and number of paper without abstract for each conference
        3. total running time
"""

import json
import numpy as numpy
import time
from datetime import timedelta

file_name = ["dblp-ref/dblp-ref-0.json","dblp-ref/dblp-ref-1.json","dblp-ref/dblp-ref-2.json","dblp-ref/dblp-ref-3.json"]
confsDict = ["Empirical Methods in Natural Language Processing", "Meeting of the Association for Computational Linguistics", "North American Chapter of the Association for Computational Linguistics"]
#confsDict = ["International Conference on Machine Learning", "Conference on Neural Information Processing Systems", "International Conference on Learning Representations","Conference on Computer Vision and Pattern Recognition", "International Conference on Computer Vision", "European Conference on Computer Vision"]
fromYear = 2000
allList = []
total_paper = [0 for i in range(len(confsDict))]
wo_abstract = [0 for i in range(len(confsDict))]


if __name__=="__main__":
    start_t = time.time()
    for file_tmp in file_name:
        # Opening JSON file
        with open(file_tmp, 'r') as handle:
            # returns JSON object as a dictionary
            data = [json.loads(line) for line in handle]

        # Iterating through the json list
        for i in data:
            print (i)
            # skip missing constrains
            if 'references' not in i.keys() or 'venue' not in i.keys():
                continue
            # check year
            year = i['year']
            if year < fromYear:
                continue
            # check conf
            conf = i['venue']
            if not any(confName.lower() in conf.lower() for confName in confsDict):
                continue
            print ("Paper:", i['id'])
            for pos_tmp,confName in enumerate(confsDict):
                if confName.lower() in conf.lower():
                    pos = pos_tmp
            total_paper[pos]+=1
            if 'abstract' not in i.keys():
                wo_abstract[pos]+=1
                abstract = "None"
            else:
                abstract = i['abstract']

            allContent = {'id':i['id'],'conf':conf,'year':year,'authors':i['authors'],'title':i['title'],'n_citation':i['n_citation'],'references':i['references'],'abstract':abstract}
            allList.append(allContent)


    # save
    with open('allDB_EMNLP+ACL+NACCL.json', 'w') as fout:
        json.dump(allList , fout, indent=4)
    # record time
    end_t = time.time()
    print ("Total time:",str(timedelta(seconds=(end_t-start_t))))
    print ("Total paper:",total_paper)
    print ("Paper without abstract:",wo_abstract)
