import pandas as pd


class DFCreationTest:
    def __init__(self):
        self.f1 = 0
        self.pre = 0
        self.su = ''
        self.res = []
        return
    def fun1(self):
        self.f1 = self.f1 + 1
        self.pre = self.pre + 100
        self.su = str(self.f1 ) + str(self.pre)
        return self.f1, self.pre,self.su
    def run(self):
        for i in range(3):
            res = self.fun1()
            self.res.append(res)
            
        df = pd.DataFrame(self.res, columns=['f1','pre','su'])
        print df.describe()
        
  
  

def main():
    test = DFCreationTest() 
    test.run()

#     print test

if __name__ == '__main__':
    main()  