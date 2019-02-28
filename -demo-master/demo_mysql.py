import pymysql
import csv
class GetData:
    def __init__(self,limit='',host='192.168.0.231',port=25686,database='kf_message',user='ylmf',password='dgMDbkpGh3t'):
        '''

        :param limit: 限制条件的列表，输入格式为例如[name=mike,age=15,id=5]转换到mysql语句为=> 'select * from table where 1=1 and name='mike' and age=15 and id=5
        :param host: mysql所在的ip地址
        :param port: mysql所在的端口号
        :param database: 访问mysql的数据库
        :param user: 访问mysql的用户名
        :param password: 访问mysql的密码
        '''
        self.limit=limit
        #连接数据库
        self.cursor = pymysql.connect(host=host, port=port, database=database,user=user,password=password,use_unicode=False).cursor()
    def __call__(self,table='chat_record_20181228', *args, **kwargs):
        '''

        :param table:需要访问的表名
        :param args:
        :param kwargs:
        :return: 返回所有遍历结果，应该是一个列表
        '''

        #拼接命令
        cmd='select * from '+table+' where 1=1'
        for i in self.limit:
            i=i.split('=')
            if i[1].isdigit():
                cmd += ' and ' + i[0] + '=' + i[1]
            else:
                cmd+=' and '+i[0]+'=\''+i[1]+'\''

        #执行语句
        self.cursor.execute(cmd)
        #返回遍历结果
        return self.cursor.fetchall()

    def save(self,data,table='chat_save_20181228'):
        '''
        :param data: 插入的数据，为一个列表，列必须与插入表中的列一一对应。
        :param table: 插入的表名
        :return:
        '''

        #拼接命令
        cmd = 'insert into '+table+' values('
        for index,d in enumerate(data):
            if index==0:
                cmd+="\'"+str(d)+"\'"
            else:
                cmd+=","+"\'"+str(d)+"\'"
        cmd+=")"

        #执行语句
        self.cursor.execute(cmd)