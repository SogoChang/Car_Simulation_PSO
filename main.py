# main.py
from RBF import RBFNetwork
from PSO import PSOTrainer
from playground import Playground
import matplotlib.pyplot as plt

def main():
    trainer = PSOTrainer(n_particles=1000, n_generations=100, rbf_hidden=6)
    result = trainer.train()

    # 不論成功與否，都要展示最佳結果
    if result and result.success:
        print("🎯 Showing successful particle behavior.")
        best_params = result.position
    elif trainer.global_best_position is not None:
        print("⚠️ No successful particle, showing best failed behavior.")
        best_params = trainer.global_best_position
    else:
        print("❌ No valid particle found.")
        return

    # 使用最佳參數建立網路
    net = RBFNetwork(n_hidden=trainer.rbf_hidden)
    net.set_parameters(best_params)

    # 模擬並顯示動作
    env = Playground()
    state = env.reset()
    fig, ax = plt.subplots()
    plt.ion()

    step = 0
    max_steps = 300
    while not env.done and step < max_steps:
        action_angle = net.forward(state)
        env.car.setWheelAngle(action_angle)
        env.car.tick()
        env._checkDoneIntersects()
        state = env.state
        env.render(ax)
        plt.pause(0.05)
        step += 1

    # 最終一幀
    env.render(ax)
    plt.draw()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
