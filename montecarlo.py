# Multiobjective Cash Management Problem using Monte Carlo methods:
# A cash manager dealing with a bank account and 2 possible transactions (increase or decrease)
# wants to minimize total costs and also cost standard deviation for an expected cash flow
# for the next 16 days. The cash management model to be used is of the Miller-Orr (1966) form
# which is based on 3 bounds low(l), high(h) and target(z).


class miller(object):
    
    def __init__(self,gzeropos,gzeroneg,gonepos,goneneg,v,u,h,z,l):
        """Initializes parameters"""
        self.gzeropos = gzeropos  #Fixed increase transaction cost 
        self.gzeroneg = gzeroneg  #Fixed decrease transaction cost
        self.gonepos = gonepos    #Variable increase transaction cost
        self.goneneg = goneneg    #Fixed decrease transaction cost
        self.v = v                #Holding cost on positive cash balances
        self.u = u                #Penalty cost on negative cash balances
        self.h = h                #High bound
        self.z = z                #Target
        self.l = l                #Low bound
        self.daily_cost = []      #List of daily costs

    def transfer(self,ob):
        """Determines transfer x from parameters, opening balance (ob) and high, z, low"""
        if ob > self.h:
            x = self.z-ob
        elif ob < self.l:
            x = self.z-ob
        else:
            x = 0
        return x
    
    def trans_cost(self,x):
        """Computes the cost of transfer x"""
        cost = 0
        if x<0:
            cost = (self.gzeroneg-self.goneneg*x)
        elif x>0:
            cost = (self.gzeropos+self.gonepos*x)
        return cost
    
    def holding_cost(self,final):
        """Computes the holding cost"""
        cost = 0
        if final>=0:
            cost = self.v*final
        else: 
            cost = -self.u*final
        return cost
                
    def cost_calc(self,cf,ob):
        """Computes a vector of daily costs for a cash flow data set and an opening balance"""
        inibal = ob
        #bal = ob
        if len(self.daily_cost)>0:
            del self.daily_cost[:]
        for element in cf:
            trans = self.transfer(inibal)
            bal = inibal+trans+element
            self.daily_cost.append(self.trans_cost(trans)+self.holding_cost(bal))
            inibal = bal
        return self.daily_cost

		
#Input data
b0 = 20
min_1 = 0
max_1 = 15
min_2 = 10
max_2 = 25
min_3 = 20
max_3 = 35
factor = 1000000
g0pos = 200
g0neg = 200
g1pos = 0.001*factor
g1neg = 0.0*factor
v = 0.001*factor
u = 0.3*factor
n = 16
flow = np.array([1,1,6,-1,-3,-3,-9,6,4,6,3,4,1,-1,-2,2],dtype=int)


#Monte Carlo simulation 
res = []
for p in range (10000):
    x1 = randint(min_1,max_1)
    x2 = randint(min_2,max_2)
    x3 = randint(min_3,max_3)
    x = [x1, x2, x3]
    if (x[0]<=x[1]) and (x[1]<=x[2]):
        model = miller(g0pos,g0neg,g1pos,g1neg,v,u,x[2],x[1],x[0])
        vcost = model.cost_calc(flow,b0)
        cost = np.mean(vcost)
        risk = np.std(vcost)
        res.append([x[0],x[1],x[2],cost,risk]) 
res = pd.DataFrame(res, columns=['l','z','h','cost','risk'])

#Auxilliary function
def front_df(S,col):
    """Accepts a data frame of policies in increasing order of cost and returns the efficient frontier"""
    P=[]
    i = 0
    P.append(S.iloc[i])
    j = 1
    while j<len(S):
        if S.iloc[j][col]< S.iloc[i][col]:
            i=j
            P.append(S.iloc[j])
        j = j+1 
    return pd.DataFrame(P)

#Returns the efficient frontier	
df = res.sort_values(by='cost')
front_df(df,col=4)