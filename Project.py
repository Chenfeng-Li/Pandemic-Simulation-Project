import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from copy import copy, deepcopy
from tqdm import tqdm


class Person:
    """
    Person object.

    position (list [x,y]): The initial position of the person. If None, then randomly assign the position.
    status (string): One of 'healthy', 'infected', 'dead' or 'recovered'. The health status of the person.
    """
    def __init__(self, position, status='healthy', mask=0, vaccinated=0):

        # One of the four possible health status

        assert status in ['healthy', 'infected', 'dead', 'recovered'], "Undefined Status"
        self.status = status
        self.position = position
        self.mask = mask # mask=0: the person is not wearing a mask; mask=1, he or she has a mask on.
        self.vaccinated = vaccinated # 0 if person is not vaccinated, 1 otherwise

        # List of status for this person, with the time changed
        self.status_list = [[status, 0]]

    def __str__(self):
        return f"{self.status} person at {self.position}"
    def __repr__(self):
        return self.status[0]

    def distance(self,other):
        """
        Compute the distance between this person with a point or another person
        other: Either a length 2 list or a Person object
        """
        if type(other)==list:
            return np.linalg.norm(np.array(self.position) - np.array(other))
        return np.linalg.norm(np.array(self.position) - np.array(other.position))
    

def virus(healthy, infected, mutation_rate=0, mask_penetration=0, vaccine_effectiveness=0):
    """
    Create a dictionary wrapping the virus property.

    healthy (function of distance, xxx): In each time step, for a healthy person,
        given the distance to an infected person, return the probability of being infected
    infected (function): In each time step, for an infected person, return the probability
        of being healed, maintain carrier, or dead
    mutation rate is the rate at which the virus mutates to another strain. Doing so renders
        previously infected people succeptible to reinfection
    """
    return {"healthy": healthy,
            "infected": infected,
            "mutation_rate": mutation_rate,
            "mask_penetration": mask_penetration,
            "vaccine_effectiveness": vaccine_effectiveness}




class Simulation:
    """
    Simulation.

    board_shape (tuple (m,n)): Shape of board.
    ts (int): Time Steps to be simulated.
    num_people (int): Number of People.
    percentage_infected (float between 0 and 1): Percentage of initial virus carriers.
    virus (dict): Property of virus.
    strategy (string): Moving strategy of the crowds.
                    "random": Random move
                    "gather": 90% to minimize the minimum distance, 10% random move
                    "stayaway": 90% to maximize the minimum distance, 10% random move
                    "party": 90% gather to the center of board, 10% random move
    seed (int): Random seed.
    """
    def __init__(self, board_shape, ts, num_people, percentage_infected,
                 virus, strategy = 'random', seed = None, percentage_masked=0, percentage_vacc=0):

        #the mutation rate is a parameter meant to be from 0 to 1 (but can technically be outside this range). The
        #higher the rate, the more likely it is that preiously infected and recovered people can be infected again

        if seed is not None:
            random.seed(seed)
        m, n = board_shape
        num_infected = int(num_people * percentage_infected)
        num_masked = int(num_people * percentage_masked)
        num_vaccinated=int(num_people * percentage_vacc)

        # List of Person objects, without assigning the position
        people = []
        for _ in range(num_people - num_infected):
            people.append(Person(position = None, status = "healthy"))
        for _ in range(num_infected):
            people.append(Person(position = None, status = "infected"))

        random.shuffle(people)

        for i in range (num_masked):
            people[i].mask=1 # gives a mask to num_masked amount of people

        random.shuffle(people) #must shuffle so same people who get vax do not get mask

        for i in range (num_vaccinated):
            people[i].vaccinated=1

        #Shuffle the list
        random.shuffle(people)

        # Assign the position for each person
        possible_position = [[i,j] for i in range(m) for j in range(n)] # All possible positions
        people_position = random.sample(possible_position, num_people)  # Sample from possible positions
        for person, person_position in zip(people, people_position):
            person.position = person_position

        # Compute the distance threshold of the virus, such that
        # Distance threshold: distance that the probability of being infected is 0.01%
        # That is, solve virus['healthy'](dist) = 0.0001
        def dist_eq(dist):
            return virus['healthy'](dist) - .0001
        initial_guess = 1
        solution = sp.optimize.fsolve(dist_eq, initial_guess)
        dist_threshold = solution[0]

        # Create the position data for KD-tree (a map people -> position)
        position_kd = {}
        for person in people:
            position_kd[person] = person.position



        # Create a board
        board = [[0 for _ in range(n)] for _ in range(m)]
        for person in people:
            person_x, person_y = person.position
            board[person_x][person_y] = person

        # Record current status
        record = {"healthy":[num_people - num_infected],
                  "infected":[num_infected],
                  "dead":[0],
                  "recovered":[0],
                  "people":[deepcopy(people)],
                  "board":[deepcopy(board)]
                 }

        self.people = people
        self.board = board
        self.board_shape = board_shape
        self.virus = virus
        self.ts = ts
        self.current_ts = 0
        self.strategy = strategy
        self.record = record
        self.dist_threshold = dist_threshold
        self.position_kd = position_kd



    def __str__(self):
        pass
    def __repr__(self):
        pass


    def move(self):
        """
        In each time step, the movement of all people.
        """
        m, n = self.board_shape

        for person in self.people:
            x, y = person.position
            possible_move = [[x,y]] # Stay
            if person.status != 'dead': # Only live person moves
                if x > 0 and self.board[x-1][y] == 0:
                    possible_move.append([x-1,y]) # Up
                if x < m-1 and self.board[x+1][y] == 0:
                    possible_move.append([x+1,y]) # Down
                if y > 0 and self.board[x][y-1] == 0:
                    possible_move.append([x,y-1]) # Left
                if y < n-1 and self.board[x][y+1] == 0:
                    possible_move.append([x,y+1]) # Right

            if self.strategy == 'random': # randomly choose next_step
                next_step = random.sample(possible_move,1)

            if self.strategy in ['gather', 'stayaway']:

                # position_kd dictionary without this person
                position_kd_wo_person = copy(self.position_kd)
                del position_kd_wo_person[person]
                position_kd_wo_person_array = np.array(list(position_kd_wo_person.values()))
                people_wo_person = list(position_kd_wo_person.keys())
                tree = sp.spatial.KDTree(position_kd_wo_person_array)

                # distance to nearest person
                dists = []
                for pm in possible_move:
                    indice = tree.query(pm)[1]
                    d = people_wo_person[indice].distance(pm)
                    dists.append(d)

                if self.strategy == 'gather':
                    # Minimize the distance to nearest person
                    next_step_idx = dists.index(min(dists))

                if self.strategy == 'stayaway':
                    # Maximize the distance to nearest perso
                    next_step_idx = dists.index(max(dists))

                if np.random.rand() < .9:    # 90% tendency
                    next_step = [possible_move[next_step_idx]]
                else:
                    next_step = random.sample(possible_move,1)

            if self.strategy == 'party':
                center = np.array([m/2, n/2])
                dists = []
                for pm in possible_move:
                    d = np.linalg.norm(np.array(pm)-center)
                    dists.append(d)
                if np.random.rand() < 0.9:
                    next_step_idx = dists.index(min(dists))
                    next_step = [possible_move[next_step_idx]]
                else:
                    next_step = random.sample(possible_move,1)

            new_x, new_y = next_step[0]
            self.board[x][y] = 0
            self.board[new_x][new_y] = person
            person.position = next_step[0]
            self.position_kd[person] = next_step[0]


    def detect(self, KD=False):
        """
        Given the current board, process detection for all people.
        """

        if not KD:
            # Infect healthy people
            for i in range(len(self.people)-1):
                for j in range(i+1,len(self.people)):
                    # Infection happens only with one healthy person and one infected person,
                    # with the infected person was sick before current time step
                    if self.people[i].status == "healthy" and self.people[j].status == "infected"\
                    and self.people[j].status_list[-1][1] != self.current_ts:
                        susceptible = self.people[i]
                    elif self.people[i].status == "infected" and self.people[j].status == "healthy"\
                    and self.people[i].status_list[-1][1] != self.current_ts:
                        susceptible = self.people[j]
                    else:
                        continue

                    dist = self.people[i].distance(self.people[j])

                    # If the susceptible person would be infected
                    infect_prob = self.virus['healthy'](dist)
                    isInfected = np.random.choice([0,1], 1, p=[1-infect_prob, infect_prob])[0]
                    if susceptible.mask == 1:
                        chance_to_prevent=np.random.rand()
                        if chance_to_prevent > self.virus["mask_penetration"]:
                            isInfected=False # when this is false, this prevents the next if statement from occuring
                    if isInfected:
                        susceptible.status = 'infected'
                        susceptible.status_list.append(['infected', self.current_ts])

        else:
            position_kd = np.array(list(self.position_kd.values()))
            tree = sp.spatial.KDTree(position_kd)

            # Infect healthy people
            for person in self.people:
                if person.status != 'infected' or person.status_list[-1][1] == self.current_ts:
                    continue
                indices = tree.query_ball_point(person.position, self.dist_threshold)

                for i in indices:
                    if self.people[i].status != 'healthy':
                        continue
                    susceptible = self.people[i]

                    dist = person.distance(susceptible)

                    infect_prob = self.virus['healthy'](dist)
                    isInfected = np.random.choice([0,1], 1, p=[1-infect_prob, infect_prob])[0]
                    if susceptible.mask == 1:
                        chance_to_prevent=np.random.rand()
                        if chance_to_prevent > self.virus["mask_penetration"]:
                            isInfected=False # when this is false, this prevents the next if statement from occuring
                    if isInfected:
                        susceptible.status = 'infected'
                        susceptible.status_list.append(['infected', self.current_ts])

        # Fate of infected people
        for person in self.people:
            # Only person infected before this time step would proceed
            if person.status == 'infected' and person.status_list[-1][1] != self.current_ts:
                next_status = self.virus['infected']()[0]
                if next_status=="dead" and person.vaccinated==1: #this will give a chance to save someone who has been vaccinated
                  random_number=np.random.random()
                  if (random_number < self.virus["vaccine_effectiveness"]): #this figure gained from this source:  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10492612/
                    person.status = next_status #in this case, they are not saved and the person dies
                  else:
                    next_status='infected' #in this case, they are saved, and the person remains infected but does not die
                else:
                  person.status = next_status
                if next_status != 'infected':
                    person.status_list.append([next_status, self.current_ts])

        ### Possible Improvement: Vaccination;
        ###                       Recovered -> Healthy (Recovered people may be infected again);
        ###                       Impact of Dead Persons


    def record_step(self):
        """
        After moving and detecting, record the status of current step.
        """
        people_status = {"healthy":0, "infected":0, "dead":0, "recovered":0}
        for person in self.people:
            people_status[person.status] += 1
        for key, val in people_status.items():
            self.record[key].append(val)

        self.record['people'].append(deepcopy(self.people))
        self.record['board'].append(deepcopy(self.board))


    def one_step(self, KD=False):
        """
        Simulate one step.
        KD: If KD, use KD-Tree in detection
        """
        self.move()
        self.detect(KD)
        self.record_step()

        if self.virus["mutation_rate"] > 0:
          for person in self.people:
            random_chance=np.random.random()
            if person.status == "recovered":
              if (random_chance < self.virus["mutation_rate"]):
                #the higher the mutation rate, the more likely a person can be infected again
                person.status = "healthy" #this person can be infected again


    def simulate(self, KD=False, verbose = True):
        """
        Simulate ts steps.
        verbose (bool): Whether to print steps of simulation
        """
        if verbose:
            iter_ts = tqdm(range(self.ts))
        else:
            iter_ts = range(self.ts)

        for _ in iter_ts:
            self.current_ts += 1
            self.one_step(KD)

    def plot(self, save_path = None):
        """
        Plot after simulation
        save_path (string): path to save the animation. e.g. "sim.png"
        """
        ts_list = list(range(self.ts+1))
        plt.plot(ts_list, self.record["healthy"], label = "healthy")
        plt.plot(ts_list, self.record["infected"], label = "infected")
        plt.plot(ts_list, self.record["recovered"], label = "recovered")
        plt.plot(ts_list, self.record["dead"], label = "dead")
        plt.xlabel("time step")
        plt.ylabel("number of person")
        plt.legend()
        plt.title("Pandemic Simulation")

        if save_path is not None:
            plt.save(save_path)

        plt.show()


    def animate(self, save_path = None):
        """
        Create Animation after simulation
        save_path (string): path to save the animation. e.g. "sim.gif"
        """
        m, n = self.board_shape
        plt.rcParams["animation.html"] = "jshtml"
        fig, ax = plt.subplots()
        scat_h = ax.scatter([], [], animated=True)
        scat_i = ax.scatter([], [], animated=True)
        scat_r = ax.scatter([], [], animated=True)
        scat_d = ax.scatter([], [], animated=True)

        def init():
            ax.grid()
            ax.set_xlim([0,n-1])
            ax.set_ylim(([0,m-1]))
            ax.set_aspect('equal', adjustable='box')
            return scat_h, scat_i, scat_r, scat_d

        def plot_step(t):
            position = {"healthy":[[],[]], "infected":[[],[]], "recovered":[[],[]], "dead":[[],[]]}
            for p in self.record["people"][t]:
                position[p.status][0].append(p.position[1])
                position[p.status][1].append(m-1-p.position[0])
            scat_h.set_offsets(np.column_stack(position["healthy"]))
            scat_i.set_offsets(np.column_stack(position["infected"]))
            scat_r.set_offsets(np.column_stack(position["recovered"]))
            scat_d.set_offsets(np.column_stack(position["dead"]))
            return scat_h, scat_i, scat_r, scat_d


        ani = animation.FuncAnimation(fig=fig, func=plot_step, frames=list(range(self.ts+1)), interval=500, init_func=init, blit=True)

        if save_path is not None:
            ani.save(save_path)
        plt.close()

        return ani