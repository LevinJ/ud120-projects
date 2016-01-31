class Base:
    def hello(self):
        print "Hello called from Base"
 
        self.hello_virtual()
 
    def hello_virtual(self):
        print "Hello called from virtual Base function"
 
class Derived(Base):
    def hello_virtual(self):
        print "Hello called from virtual Derived function"
 
d = Derived()
d.hello()