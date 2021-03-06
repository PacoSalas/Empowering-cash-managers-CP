# Multiobjective Cash Management Problem using Linear Programming:
# A cash manager dealing with a system with 2 bank accounts and 2 possible transactions
# wants to minimize total costs and also Conditional Cost at Risk for an expected cash flow
# for the next 16 days. The model include minimum balance constraints and external net cash 
# flow forecasts for each account for each time period.


from gurobipy import *

# Time periods
n = 16
time_range = range(n)

# Transactions
trans = [1,2]
tr_range = range(len(trans))

# Bank accounts
banks = [1,2]
bk_range = range(len(banks))

# Trans fixed costs
g0 = {1:200, 2:200}
vg0 = np.array([200,200]).reshape((2,1))

# Trans variable costs
g1 = {1:1000, 2:0.0}
vg1 = np.array([1000,0]).reshape((2,1))

# Initial balance
b0 = [20,0]

# Minimum balances
bmin = [7,-100]

# Holding costs per bank account
v = {1:1000, 2:0.0}
vv = np.array([1000,0]).reshape((2,1))

# Expected external net flow
f = [[1,0],[1,0],[6,0],[-1,0],[-3,0],[-3,0],[-9,0],[6,0],[4,0],[6,0],[3,0],[4,0],[1,0],[-1,0],[-2,0],[2,0]]

# Allowed transactions between accounts
# e.g. "[1, -1" means that account 1 adds trans1 minus trans2
A =[[1, -1], [-1, 1]]

# Weights
w1 = 0.5
w2 = 1 - w1

# Cost and risk bounds
Cmax = 150000
Rmax = 150000

#Cost reference
c0 = 2000

# Model
m = Model("lp")

# Fixed costs: z = 1 if trans x occurs at time tau
fixed = []
for tau in time_range:
    fixed.append([])
    for t in trans:
        fixed[tau].append(m.addVar(vtype=GRB.BINARY, name="z%d,%d" %(tau,t)))
m.update()

# Variable costs are proportional to transaction decision variables
var = []
for tau in time_range:
    var.append([])
    for t in trans:
        var[tau].append(m.addVar(vtype=GRB.CONTINUOUS, name="x%d,%d" %(tau,t)))
m.update()

# Holding costs are proportional to balance auxiliary decision variables
bal = []
for tau in time_range:
    bal.append([])
    for j in banks:
        bal[tau].append(m.addVar(vtype=GRB.CONTINUOUS, name="b%d,%d" %(tau,j)))
m.update()

# Deviational variables above a given cost
devpos = []
devneg = []
for tau in time_range:
    devpos.append(m.addVar(vtype=GRB.CONTINUOUS, name="devpos%d" %tau))
    devneg.append(m.addVar(vtype=GRB.CONTINUOUS, name="devneg%d" %tau))
m.update()

# Deviation constraints 
tc = []
hc = []
for tau in time_range:
    tc.append(sum([g0[t]*fixed[tau][t-1] + g1[t]*var[tau][t-1] for t in trans]))
    hc.append(sum([v[j]*bal[tau][j-1] for j in banks]))
    m.addConstr(tc[tau] + hc[tau] - devpos[tau] + devneg[tau] == c0, 'DevCon%s'%tau)
              
# Intitial transition constraints and minimum balance constraints
for j in bk_range:
    m.addConstr(b0[j] + f[0][j] + LinExpr(A[j],var[0][:]) == bal[0][j], 'IniBal%d'% j)
    m.addConstr(bal[0][j] >= bmin[j], 'Bmin%s'%j)
m.update()

# Rest of transition constraints
for tau in range(1,n):
    for j in bk_range:
        m.addConstr(bal[tau-1][j] + f[tau][j] + LinExpr(A[j],var[tau][:]) == bal[tau][j], 'Bal%d,%d'%(tau,j))
        m.addConstr(bal[tau][j] >= bmin[j], 'Bmin%d%d'%(tau,j))
m.update()

# Exclusivity constraints
for tau in time_range:
    m.addConstr(fixed[tau][0] + fixed[tau][1] <= 1, name="ex%d" %tau)
m.update() 


# Bounds and binary variables constraints
for tau in time_range:
    for i in tr_range:
        m.addConstr(var[tau][i] <= 1000*fixed[tau][i], name="c%d%d" %(tau,2*i+1))
        m.addConstr(var[tau][i] >= 0.001*fixed[tau][i], name="c%d%d" %(tau,2*i+2))
m.update() 

              
# Setting the objectives
transcost = sum([g0[t]*fixed[tau][t-1] + g1[t]*var[tau][t-1] for tau in range(n) for t in trans])
holdcost = sum([v[j]*bal[tau][j-1] for tau in range(n) for j in banks])
devcost = sum([devpos[tau] for tau in range(n)])
m.setObjective((w1/Cmax)*(transcost+holdcost)+(w2/Rmax)*devcost,GRB.MINIMIZE)
m.update()

# Budget constraints
m.addConstr(transcost + holdcost <= Cmax, name="CostBudget")
m.addConstr(devcost <= Rmax, name="RiskBudget")
m.update()

# Optimization
m.optimize()

# Print policy and balance solution
resx= []
resb= []
for dv in m.getVars():
    if 'x' in dv.varName:
        resx.append([dv.varName, dv.x])
    if 'b' in dv.varName:
        resb.append([dv.varName, dv.x])    
print('Obj: %g' % m.objVal)