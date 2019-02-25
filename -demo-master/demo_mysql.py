import pymysql
import csv
class get_data:
    def __init__(self,limit='',host='192.168.0.231',port=25686,database='kf_message',user='ylmf',password='dgMDbkpGh3t'):
        self.limit=limit
        self.cursor = pymysql.connect(host=host, port=port, database=database,user=user,password=password,use_unicode=False).cursor()
    def __call__(self,date=20181228, *args, **kwargs):
        cmd='select * from chat_record_'+str(date)+' where 1=1'
        for i in self.limit:
            cmd+=' and '+i
        # cmd+=" limit 10"
        # print(cmd)
        self.cursor.execute(cmd)
        # cursor.close()
        return self.cursor.fetchall()
    def save(self,data,date):
        cmd = 'insert into chat_save_'+str(date)+' values('
        for index,d in enumerate(data):
            if index==0:
                cmd+="\'"+str(d)+"\'"
            else:
                cmd+=","+"\'"+str(d)+"\'"
        cmd+=")"
        # print(cmd)
        # exit(0)
        self.cursor.execute(cmd)

def ttt():
    # x=get_data()
    # x.save([1,2,3,4,5])
    x=get_data()
    # file = open('../data/chat/all_data.csv', 'w',encoding='utf-8')
    with open('../data/all_data121.csv','w',encoding='utf-8')as w:
        for i in range(20181215,20181232):
            for data in x(i):
                try:
                    w.write(data[11].decode('utf-8') + "," + str(data[10]) + '\n')
                except:
                    continue
                # exit(0)
        print('11111')
    w.close()
    # with open('../data/all_data02.csv', 'w')as w:
    #     for i in range(20190201,20190217):
    #         for data in x(i):
    #             try:
    #                 w.write(data[11].decode('utf-8') + "," + str(data[10]) + '\n')
    #             except:
    #                 continue
    #     print('33333')
    # w.close()
    exit(0)
    print('finish.')
    with open('../data/all_data01.csv','w')as w:
        for i in range(20190101,20190132):
            for data in x(i):
                try:
                    w.write(data[11].decode('utf-8') + "," + str(data[10]) + '\n')
                except:
                    continue
        print('22222')
    w.close()

    exit(0)
def label_data():
    keywords = set()
    with open('../data/chat/keywords.txt', 'r', encoding='utf-8')as f:
        d = f.readline().strip()
        while d:
            keywords.add(d)
            d = f.readline().strip()
    f.close()
    # print(len(keywords))
    # exit(0)
    x = get_data()
    with open('../data/all_data121.csv','w',encoding='utf-8')as w:
        for i in range(20181215,20181232):
            for data in x(i):
                try:
                    s=data[11].decode('utf-8')
                    for k in keywords:
                        if k in s:
                            print(s)
                            w.write(s + "," + str(data[10]) + '\n')
                            break
                except:
                    continue
                # exit(0)
        print('11111')
    w.close()
    with open('../data/all_data021.csv', 'w')as w:
        for i in range(20190201,20190217):
            for data in x(i):
                try:
                    s=data[11].decode('utf-8')
                    for k in keywords:
                        if k in s:
                            print(s)
                            w.write(s + "," + str(data[10]) + '\n')
                            break
                except:
                    continue
        print('33333')
    w.close()
    # exit(0)
    print('finish.')
    with open('../data/all_data011.csv','w')as w:
        for i in range(20190101,20190132):
            for data in x(i):
                try:
                    s=data[11].decode('utf-8')
                    for k in keywords:
                        if k in s:
                            print(s)
                            w.write(s + "," + str(data[10]) + '\n')
                            break
                except:
                    continue
        print('22222')
    w.close()

label_data()