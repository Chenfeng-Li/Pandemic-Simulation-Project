from Project import Person, Simulation, virus
import unittest
import numpy as np

class TestPerson(unittest.TestCase):
    """
    Test the class Person
    """
    def test_person_construction(self):
        # Check if the person is initiated with the correct properties
        person = Person(position=[1,1], status='healthy', mask=1)
        self.assertEqual(person.status, 'healthy')
        self.assertEqual(person.position, [1,1])
        self.assertEqual(person.mask,1)
        self.assertEqual(person.status_list, [['healthy', 0]])

    def test_person_status_change(self):
        # Test if the status of a person changes correctly
        person = Person(position=[0, 0], status='healthy', mask=1)
        self.assertEqual(person.status, 'healthy')
        person.status = 'infected'
        self.assertEqual(person.status, 'infected')

    def test_person_movement(self):
        # Test if the movement of a person changes correctly
        person = Person(position=[0, 0], status='healthy', mask=1)
        person.position = ([1, 1])
        self.assertEqual(person.position, [1, 1])

    def test_distance_person(self):
        # Test if the movement of a person changes correctly
        person = Person(position=[1,1], status='healthy', mask=1)
        another_person = Person([0, 0])
        distance = person.distance(another_person)
        self.assertAlmostEqual(distance, 1.41, places=2)

class TestSimulation(unittest.TestCase):
    """
    Test the class Simulation
    """
    # Initialize the values for parameters
    board_shape=(5, 5)
    ts=100
    num_people=10
    percentage_infected = 0.5
    seed=123
    virus = virus(lambda dist: 0.9/dist**2, lambda: np.random.choice(["infected", "dead", "recovered"],size=1, p=[0.95, 0.005, 0.045]))
    strategy = 'random'

    def test_simulation_construction(self):
        """
        Check if the simulation is initiated with the correct properties
        """
        np.random.seed(self.seed)
        simulation_1 = Simulation(board_shape=self.board_shape, ts=self.ts, num_people=self.num_people,
                                percentage_infected=self.percentage_infected, virus=self.virus,
                                strategy=self.strategy, seed=self.seed)

        np.random.seed(self.seed + 1)
        simulation_2 = Simulation(board_shape=self.board_shape, ts=self.ts, num_people=self.num_people,
                                percentage_infected=self.percentage_infected, virus=self.virus,
                                strategy=self.strategy, seed=self.seed+1)

        # Check if properties are consistent with the initialized values
        self.assertEqual(simulation_1.board_shape, self.board_shape)
        self.assertEqual(simulation_1.ts, self.ts)
        self.assertEqual(len(simulation_1.people), self.num_people)
        self.assertEqual(simulation_1.current_ts, 0)


        # Check if randomness works in initializing positions
        position_1 = [person.position for person in simulation_1.people]
        position_2 = [person.position for person in simulation_2.people]
        self.assertNotEqual(position_1, position_2)

        # Check if randomness works in initializing status
        status_1 = {
            'healthy': sum(person.status=='healthy' for person in simulation_1.people),
            'infected': sum(person.status=='infected' for person in simulation_1.people),
            'dead': sum(person.status=='dead' for person in simulation_1.people),
            'recovered': sum(person.status=='recovered' for person in simulation_1.people)
            }
        expected_infected_count = int(self.num_people * self.percentage_infected)
        self.assertAlmostEqual(status_1['infected'], expected_infected_count, delta=1)
        self.assertAlmostEqual(status_1['healthy'], self.num_people - expected_infected_count, delta=1)


    def test_simulation_movement(self):
        """
        Test if people in the simulation move correctly
        """
        simulation = Simulation(board_shape=self.board_shape, ts=self.ts, num_people=self.num_people, percentage_infected=self.percentage_infected, virus=self.virus, strategy=self.strategy, seed=self.seed)
        initial_position = [person.position for person in simulation.people]
        simulation.move()
        updated_position = [person.position for person in simulation.people]
        self.assertNotEqual(initial_position, updated_position)

        #  Further assertion
        # Check if the new positions are within the valid dimension of the board
        m, n = self.board_shape
        for position in updated_position:
            self.assertTrue(0 <= position[0] < m, "A person moved out of the board in the vertical direction.")
            self.assertTrue(0 <= position[1] < n, "A person moved out of the board in the horizontal direction.")

        # Check if people are not overlapping
        self.assertEqual(len(set(map(tuple, updated_position))), len(updated_position), "People moved to the same position.")


    def test_record_step(self):
        """
        Check if the record step captures each step
        """
        simulation = Simulation(board_shape=self.board_shape, ts=self.ts, num_people=self.num_people, percentage_infected=self.percentage_infected, virus=self.virus, strategy=self.strategy, seed=self.seed)

        # Simulate a few steps
        for _ in range(5):
            simulation.move()
            simulation.detect()
            simulation.record_step()

        # Check the length of recorded status for each step
        recorded_status = simulation.record
        self.assertEqual(len(recorded_status['healthy']), 6)
        self.assertEqual(len(recorded_status['infected']), 6)
        self.assertEqual(len(recorded_status['recovered']), 6)
        self.assertEqual(len(recorded_status['dead']), 6)
        self.assertEqual(len(recorded_status['people']), 6)
        self.assertEqual(len(recorded_status['board']), 6)