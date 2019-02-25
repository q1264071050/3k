# !/usr/bin/python
# -*- coding: utf-8 -*-
import tkinter
import os
class FirstWindow(object):
    def __init__(self,base_dir,data_list):
        self.top = tkinter.Tk()
        self.base_dir=base_dir
        self.scrollbar = tkinter.Scrollbar(self.top)
        self.scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.text = tkinter.Text(self.top,
                                 width=100,
                                 height=40,
                                 font=("Times New Roman", 14),
                                 yscrollcommand=self.scrollbar.set)
        self.text.pack()
        self.text2 = tkinter.Text(self.top,
                                 width=8,
                                 height=1,
                                 font=("Times New Roman", 14))
        self.text2.pack(side=tkinter.LEFT)
        self.scrollbar.config(command=self.text.yview)

        self.button = tkinter.Button(text="是",
                                     command=self.is_one)
        self.button.pack()
        self.button2 = tkinter.Button(text="不是",
                                     command=self.is_zero)
        self.button2.pack()
        # self.button2 = tkinter.Button(text="保存",
        #                               command=self.save)
        # self.button2.pack()
        self.data_list=data_list
        self.begin=0
        for i in range(len(data_list)):
            if data_list[i][1]==-1:
                self.begin=i
                break
        self.show(self.data_list[self.begin][0])
        self.text2.insert(tkinter.END,str(self.begin)+"/"+str(len(data_list)))
    def show(self,name):
        f = open(self.base_dir + name, 'r', encoding='utf-8')
        d = f.readline()
        num = 0
        while d:
            if num > 500:
                break
            try:
                num += 1
                self.text.insert(tkinter.END, d)
            except:
                continue
            d = f.readline()
        f.close()
    def is_one(self):
        #改标签
        self.data_list[self.begin][1]=1
        #下一个类
        self.begin+=1
        #清空
        self.text.delete('1.0', 'end')
        self.text2.delete('1.0', 'end')
        self.text2.insert(tkinter.END, str(self.begin) + "/" + str(len(self.data_list)))
        #读内容
        self.show(self.data_list[self.begin][0])
        w = open(self.base_dir + 'label.csv', 'w', encoding='utf-8')
        for index in range(len(self.data_list)):
            w.write(self.data_list[index][0] + ',' + str(self.data_list[index][1]) + '\n')
        w.close()
    def is_zero(self):
        # 改标签
        self.data_list[self.begin][1] = 0
        # 下一个类
        self.begin += 1
        # 清空
        self.text.delete('1.0', 'end')
        self.text2.delete('1.0', 'end')
        self.text2.insert(tkinter.END, str(self.begin) + "/" + str(len(self.data_list)))

        # 读内容
        self.show(self.data_list[self.begin][0])
        w = open(self.base_dir + 'label.csv', 'w', encoding='utf-8')
        for index in range(len(self.data_list)):
            w.write(self.data_list[index][0] + ',' + str(self.data_list[index][1]) + '\n')
        w.close()
    def save(self):
        w = open(self.base_dir + 'label.csv', 'w', encoding='utf-8')
        for index in range(len(self.data_list)):
            w.write(self.data_list[index][0] + ',' + str(self.data_list[index][1]) + '\n')
        w.close()

def MakeLabel(dir='/home/second/PycharmProjects/aaa/chat_demo/cluster/'):
    files=os.listdir(dir)
    data_list=[]
    if 'label.csv' not in files:
        w=open(dir+'label.csv','w',encoding='utf-8')
        for file in files:
            w.write(file+','+'-1'+'\n')
            data_list.append([file,-1])
        w.close()
    else:
        r = open(dir + '/label.csv', 'r', encoding='utf-8')
        d=r.readline().strip()
        while d:
            d=d.split(',')
            data_list.append([d[0],int(d[1])])
            d=r.readline().strip()
        r.close()
    # print(data_list)
    # print(len(data_list))
    # exit(0)
    FirstWindow(dir,data_list)
    tkinter.mainloop()


if __name__ == '__main__':
    MakeLabel()
