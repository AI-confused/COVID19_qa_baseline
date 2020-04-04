import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import argparse



def dispose_train(train_df, passage_df, clean_train_dir):
    s,e,a = [], [], []
    examples=[]
    for i,val in enumerate(train_df[['id','docid','question','answer']].values):
        passage = passage_df[passage_df['docid']==val[1]].values[0][-1]
        ans_l = len(val[-1])
        start = passage.find(val[-1])
        end = start+ans_l
        s.append(start)
        e.append(end)  
        #如果训练集中存在答案不能和文档匹配情况，去除该训练样本
        if val[-1]!=passage[start:end]:
            a.append(i)
                    
    train_df['start_index']=s
    train_df['end_index']=e
    train_df.drop(a,inplace=True)
    train_df.to_csv(clean_train_dir,index=False, header=True)


def clean_data(train_dir, clean_train_dir, passage_dir, clean_passage_dir, test_dir, clean_test_dir):
    #clean passage file
    passage = pd.read_csv(passage_dir)
    for i in passage.index:
        passage.iloc[i,-1] = passage.iloc[i,-1].replace(' ', '').replace('\t', '')

    passage.to_csv(clean_passage_dir, index=False,header=True)
    #clean train file
    train = pd.read_csv(train_dir, sep='\t')
    for i in train.index:
        train.iloc[i,2] = train.iloc[i,2].replace(' ','').replace('\t', '')
        train.iloc[i,3] = train.iloc[i,3].replace(' ','').replace('\t', '')
    
    dispose_train(train, passage, clean_train_dir)
    #clean test file
    test = pd.read_csv(test_dir)
    for i in test.index:
        test.iloc[i,-1] = test.iloc[i,-1].replace(' ','').replace('\t', '')

    test.to_csv(clean_test_dir, index=False,header=True)
    print('clean done')


class ElasticObj:
    def __init__(self, index_name,index_type,ip ="127.0.0.1"):
        '''
        构建es索引，批量导入数据
        '''
        self.index_name =index_name
        self.index_type = index_type
        self.es = Elasticsearch([ip])

    def bulk_Index_Data(self, csvfile):
        '''
        用bulk将批量数据存储到es
        '''
        df = pd.read_csv(csvfile)
        doc = []
        for item in df.values:
            dic = {}
            dic['docid'] = item[0]
            dic['passage'] = item[1]
            doc.append(dic)
        ACTIONS = []
        i = 0
        for line in doc:
            action = {
                "_index": self.index_name,
                "_type": self.index_type,
                "_source": {
                    "docid": line['docid'],
                    "passage": line['passage']}
            }
            i += 1
            ACTIONS.append(action)
        print('index_num:',i)
        success, _ = bulk(self.es, ACTIONS, index=self.index_name, raise_on_error=True)
        print('Performed %d actions' % success)
        

    def create_index(self,index_name,index_type):
        '''
        创建索引
        '''
        #创建映射
        _index_mappings = {
            "mappings": {
                    "properties":{
                          "passage":{
                            "type":"text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_max_word"
                          },
                          "docid":{
                            "type": "text"
                          }
                        }
            }
        }
        #构建索引
        if self.es.indices.exists(index=self.index_name) is not True:
            res = self.es.indices.create(index=self.index_name, body=_index_mappings)
            print(res)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--passage_dir", default=None, type=str, required=True,
                        help="The passage data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--test_dir", default=None, type=str, required=True,
                        help="The test data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--clean_train_dir", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--clean_passage_dir", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--clean_test_dir", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--es_index", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--es_ip", default=None, type=str, required=True,
                        help="")
    args = parser.parse_args()
    clean_data(args.train_dir, args.clean_train_dir, args.passage_dir, args.clean_passage_dir, args.test_dir, args.clean_test_dir)
    #建立ES，把文档批量导入索引节点
    obj = ElasticObj(args.es_index,"_doc",ip =args.es_ip)
    obj.create_index(args.es_index,"_doc")
    obj.bulk_Index_Data(args.clean_passage_dir)