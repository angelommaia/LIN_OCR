from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, LpMaximize
from pulp import GLPK

model = LpProblem("Coordenacao_Linear", LpMinimize)

quantidade_reles = 3

#constantes


#variaveis de decisao
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, quantidade_reles)}


#restricoes
model += (3.28*x[1] - 3.28*x[2] >= 0.5,"CTI3")
model += (4.2790*x[1] >= 0.5,"Tempo de Operacao R1")
model += (4.9790*x[2] >= 0.5,"Tempo de Operacao R2")

#objetivo
model += 2.87*x[1] + 3.28*x[2]

# Solve the optimization problem
status = model.solve()

# Get the results
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")

for var in x.values():
    print(f"{var.name}: {var.value()}")

for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")







#model = pulp.LpProblem("Coordenacao_Linear", pulp.LpMinimize)


# x1 = pulp.LpVariable("TDS1", lowBound=0, cat="Continuous")
# x2 = pulp.LpVariable("TDS2", lowBound=0, cat="Continuous")
# x3 = pulp.LpVariable("TDS3", lowBound=0, cat="Continuous")


# coordenacao_linear += 4.279*x1 + 4.979*x2 + 6.3019*x3

# coordenacao_linear += 6.3019*x2 - 6.3029*x3 >= 0.3

# coordenacao_linear += 6.3019*x1 - 6.3029*x2 >= 0.3

# coordenacao_linear += 4.9790*x1 - 4.9790*x2 >= 0.3

# coordenacao_linear += 4.2790*x1 >= 0.2

# coordenacao_linear += 4.9790*x2 >= 0.2

# coordenacao_linear += 6.3019*x3 >= 0.2


# coordenacao_linear.solve()

# coordenacao_linear.solve()
# pulp.LpStatus[coordenacao_linear.status]

# for variable in coordenacao_linear.variables():
	# print("{} = {}".format(variable.name, variable.varValue))

