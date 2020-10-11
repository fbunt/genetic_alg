import numpy as np
import matplotlib.pyplot as plt


class Problem:
    """
        A problem represents a m-dim knapsack problem with m knapsacks, each bound by constraint b_i for i = 1, ..., m
        and n items each with profit p_j and cost r_ij for j = 1, ...., n

        Profits, weights, and bag limits each given by the Chu and Beasley paper

        self.r is an m x n np array
        self.p is a  1 x n np array
        self.b is a  1 x m np array
    """

    def __init__(self, n, m, alpha):
        self.items = n
        self.knapsacks = m
        # Generate the weight (restraint) for i = 1, ..., m knapsacks with j = 1, ..., n items
        self.r = np.random.randint(low=0, high=1000, size=(m, n))
        # Generate bag constraint b_i
        self.b = np.zeros(m)
        self.b = alpha * np.sum(self.r, axis=1)
        self.b = self.b.astype(int)
        # Generate the profit for j = 1, ..., n items
        q = np.random.rand(1, n)
        self.p = np.sum(np.multiply(self.r, 1 / m), axis=0) * q
        self.p = self.p.astype(int)


def initialize_pop(N, ga_problem):
    """
    This function implements Algorithm 2, Initialize P(0) for the MKP
    s.t. P(0) = {S_1, ... , S_N} where S_i in {0, 1}^n

    Each of the initial feasible solutions randomly selects a variable, and sets it to one if the solution is feasible
    :param ga_problem:
    :param N: number of chromosomes to appear in the solution space S
    :return: returns a population, pop, with N chromosomes each reflecting the selection of n items
    """
    # initialize the N x n population with everything set to 0
    number_of_items = ga_problem.items
    number_of_bags = ga_problem.knapsacks
    S = np.zeros(shape=(N, number_of_items), dtype=int)
    # Loop through each k = 1, ..., N possible solutions and randomly add item i to knapsack j,
    # UNLESS doing so violates the r_ij * x_j <= b_i condition
    for k in range(0, N):
        # generate list T of all possible j = 1, ..., n items that can exist in S_k
        T = np.arange(number_of_items)
        np.random.shuffle(T)
        # pop element j from T
        j, T = T[-1], T[:-1]
        for i in range(0, number_of_bags):
            R = 0
            while R + ga_problem.r[i, j] <= ga_problem.b[i]:
                S[k, j] = 1
                R += ga_problem.r[i, j]
                j, T = T[-1], T[:-1]

    return S


def evaluate_fitness(population, problem):
    number_solutions = population[:, 0].size
    # initialize fitness function vector
    evaluated_fitness = np.zeros(number_solutions)
    # calculate fitness
    evaluated_fitness[:] = np.sum(np.multiply(problem.p, population[:, :]), axis=1)
    return evaluated_fitness


def binary_tournament_selection(population, tournament_fitness):
    """
    This function performs a binary tournament selection from individuals randomly selected in the population
    Since this is a BINARY tournament selection we pick T = 2 meaning that we randomly select four members of the
    population and put them in sets T1 and T2. The best element in T1 becomes P1, and the best element in T2 becomes
    P2
    :param population:
    :param tournament_fitness:
    :return: parent_1, parent_2 are the ROW indices of the parents 1 and 2 in the population vectors
    """
    # T = 2 was chosen by the authors
    T = 2
    # generate the population pool
    population_pool = np.arange(population[:, 0].size)
    # pick 2 items for T_1
    T_1 = np.random.choice(population_pool, T, replace=False)
    population_pool = np.delete(population_pool, T_1)
    T_2 = np.random.choice(population_pool, T, replace=False)
    # parent_1 is winner of T_1
    parent_1 = T_1[np.argmax(tournament_fitness[T_1])]
    # parent_2 is winner of T_2
    parent_2 = T_2[np.argmax(tournament_fitness[T_2])]

    return parent_1, parent_2


def uniform_crossover(parent_1, parent_2):
    """
    This function applies the uniform crossover function by iterating through all the i bits in the two parents
    a bit b = [0, 1] is randomly selected. For all the bits in the child C, if b = 0 then the ith bit of C is taken
    from parent_1, if b = 1, then the ith bit of C is taken from parent_2
    :param parent_1:
    :param parent_2:
    :return:
    """
    # initialize the child C
    C = np.zeros(parent_1.size, dtype=int)

    # loop through the bits of the parents
    for i in range(parent_1.size):
        b = np.random.randint(2, size=1)
        # if b = 0 make the ith bit of C = ith bit of parent_1
        if b == 0:
            C[i] = parent_1[i]
        # if b = 1 make the ith bit of C = ith bit of parent_2
        else:
            C[i] = parent_2[i]
    return C


def mutate(child):
    """
    This function applies a mutator operation to C.
    Mutator operation: with 50% probability, the operator will flip  a random bit of C
    :param child:
    :return: C: the mutated child
    """
    mutation = np.random.randint(2, size=1)
    # don't apply a mutation
    if mutation == 1:
        return child
    # apply a mutation
    else:
        # pick a bit location to flip
        bit_pool = np.arange(child[:].size)
        i = np.random.choice(bit_pool, 1, replace=False)
        # if child[i] is a 0 make it a 1
        if child[i] == 0:
            child[i] = 1
        else:
            child[i] = 0
        return child


def repair(S, problem, operator):
    """
    This function implements two different repair types to the enforce the resource constraints of the bags
    1) Implements Algorithm 1 the "fancy" repair operator
    2) Sets
    :param S:
    :param problem:
    :param operator:
    :return:
    """
    # Initialize R
    R = np.zeros(problem.knapsacks, dtype=int)
    R[:] = np.sum(np.multiply(problem.r[:, :], S[:]), axis=1)

    if operator == "normal":
        if np.any(R) > np.any(problem.b):
            S = np.zeros(S.size, dtype=int)
        return S

    # This implements the fancy repair operation
    if operator == "fancy":
        pass
        # DROP phase
        for j in range(problem.items):
            if (S[j] == 1) and (np.all(R) > np.all(problem.b)):
                S[j] = 0
                R[:] = R[:] - problem.r[:, j]


def find_ga(k, total_iterations, problem, repair_operator):
    """
    Function implements Algorithm 3: a GA for the MKP from the Chu and Beasley paper
    :param k:
    :param total_iterations:
    :param problem:
    :param repair_operator:

    :return:
    """
    t = 0
    # Initialize population P(0) = {S_1, ..., S_N}, S_i in {0, 1}^n
    population = initialize_pop(k, problem)
    # Evaluate P(0) = {f(S_1), ..., f(S_N)}
    fitness = evaluate_fitness(population, problem)
    # find S* in P(0) s.t. F(S*) > f(S) for all S in P(0).
    # i.e. find the best member of the population
    max_fitness_index = np.argmax(fitness)
    solution_record = np.zeros((total_iterations, problem.items))
    fitness_record = np.zeros((total_iterations, 1))
    fitness_record[t, 0] = fitness[max_fitness_index]
    solution_record[t, :] = population[max_fitness_index]
    print("So the best guess is ", solution_record[t, :], " with value ", fitness_record[t, 0])
    t += 1

    # now we begin our iterations
    while t < total_iterations:
        # carry over the record book from the previous time step
        # update it with the child at the end of the while loop if appropriate
        fitness_record[t, 0] = fitness_record[t-1, 0]
        solution_record[t, :] = solution_record[t-1, :]
        # select parents 1 and 2 {P_1, P_2} = phi(P(t)) where phi is our binary tournament selection
        parent_1, parent_2 = binary_tournament_selection(population, fitness)
        # Crossover C = omega(P_1, P_2) where omega is our uniform crossover operation
        C = uniform_crossover(population[parent_1], population[parent_2])
        # Mutate C with our mutation operator
        C = mutate(C)
        # Make C feasible by applying the repair operator
        C = repair(C, problem, repair_operator)
        # Make sure there are no duplicate children
        # while C in population:
        #    parent_1, parent_2 = binary_tournament_selection(population, fitness)
        #    C = uniform_crossover(population[parent_1], population[parent_2])
        #    C = mutate(C)
        #    C = repair(C, problem, repair_type)
        # evaluate f(C)
        C_fitness = np.sum(np.multiply(problem.p[:], C[:]))
        # C_fitness = evaluate_fitness(C, problem)
        # Find member of the population with the lowest fitness and replace it with C
        S_prime = np.argmin(fitness[:])
        if C_fitness > fitness[S_prime]:
            population[S_prime] = C
        # Check of C is the best solution
        if C_fitness > fitness_record[t, 0]:
            fitness_record[t, 0] = C_fitness
            solution_record[t, :] = C
        t += 1

    return fitness_record, solution_record


if __name__ == '__main__':
    # Set parameters
    t_max = 1000
    pop_size = 10
    items = 25
    bags = 3
    tightness_ratio = .1
    repair_type = "normal"

    problem_1 = Problem(items, bags, tightness_ratio)
    fitness_final, solution_final = find_ga(pop_size, t_max, problem_1, repair_type)
    print("And after ", t_max, " iterations our best guess is ", solution_final[999, :],
          " with value ", fitness_final[-1, 0])

    plt.plot(fitness_final)
    plt.ylabel('fitness')
    plt.show()
