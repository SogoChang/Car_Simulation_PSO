# pso_trainer.py
import numpy as np
from RBF import RBFNetwork
from playground import Playground
from simple_geometry import Point2D

class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(-1, 1, dim)
        self.velocity = np.zeros(dim)
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')
        self.success = False  # 是否成功到達終點

class PSOTrainer:
    def __init__(self, n_particles=100, n_generations=50, rbf_hidden=6):
        self.n_particles = n_particles
        self.n_generations = n_generations
        self.rbf_hidden = rbf_hidden
        self.network_dim = RBFNetwork(n_hidden=rbf_hidden).get_parameter_size()
        self.particles = [Particle(self.network_dim) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.success_particle = None  # 儲存成功的粒子

    def simulate(self, net: RBFNetwork):
        env = Playground()
        state = env.reset()
        steps = 0
        max_steps = 300

        while not env.done and steps < max_steps:
            action_angle = net.forward(state)
            env.car.setWheelAngle(action_angle)
            env.car.tick()
            env._checkDoneIntersects()
            state = env.state
            steps += 1

        if env.reached_goal:
            return 0.0, True  # 成功抵達，fitness 最小
        else:
            # 撞牆：回傳車體中心與終點中心的距離
            car_final_pos = env.car.getPosition('center')
            dest_center = Point2D(
                (env.destination_topleft.x + env.destination_bottomright.x) / 2,
                (env.destination_topleft.y + env.destination_bottomright.y) / 2
            )
            distance = car_final_pos.distToPoint2D(dest_center)
            return distance, False


    def train(self):
        for gen in range(self.n_generations):
            print(f"\n🌀 Generation {gen}")
            for particle in self.particles:
                net = RBFNetwork(n_hidden=self.rbf_hidden)
                net.set_parameters(particle.position)
                fitness, success = self.simulate(net)

                if success:
                    particle.success = True
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = particle.position.copy()
                        self.success_particle = particle
                    print(f"✅ Particle reached destination in {fitness} steps!")
                    # 成功直接 early stop
                    return self.success_particle

                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            # 更新每個粒子
            for particle in self.particles:
                inertia = 0.5
                cognitive = 1.5
                social = 1.5
                r1, r2 = np.random.rand(), np.random.rand()
                particle.velocity = (
                    inertia * particle.velocity +
                    cognitive * r1 * (particle.best_position - particle.position) +
                    social * r2 * (self.global_best_position - particle.position)
                )
                particle.position += particle.velocity

            print(f"📉 Best fitness this gen: {self.global_best_fitness:.2f}")

        print("❌ No particle reached destination after all generations.")
        return self.success_particle if self.success_particle else None
