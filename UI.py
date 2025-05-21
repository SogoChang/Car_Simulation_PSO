import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from RBF import RBFNetwork
from PSO import PSOTrainer
from playground import Playground
import sys

class PSOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PSO-RBF 車輛訓練模擬")

        self.params_frame = ttk.LabelFrame(root, text="參數設定")
        self.params_frame.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

        self._add_param("粒子數", "150")
        self._add_param("最大代數", "50")
        self._add_param("RBF 隱藏層", "6")

        self.buttons_frame = ttk.Frame(root)
        self.buttons_frame.grid(row=1, column=0, sticky="sw", padx=10)

        ttk.Button(self.buttons_frame, text="開始訓練", command=self.start_training).grid(row=0, column=0, padx=5, pady=5)
        self.eval_button = ttk.Button(self.buttons_frame, text="Start Evaluation", command=self.start_evaluation, state=tk.DISABLED)
        self.eval_button.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.buttons_frame, text="重置", command=self.reset_all).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(self.buttons_frame, text="結束", command=self.exit_program).grid(row=0, column=3, padx=5, pady=5)
        self.pause_button = ttk.Button(self.buttons_frame, text="暫停訓練", command=self.toggle_pause)
        self.pause_button.grid(row=0, column=4, padx=5, pady=5)

        self.text_var = tk.StringVar(value="Generation: 0 | Fitness: ---")
        self.status_label = ttk.Label(root, textvariable=self.text_var, font=("Arial", 12))
        self.status_label.grid(row=2, column=1, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=2)

        self.trainer = None
        self.best_particle = None
        self.paused = False
        self.interrupted = False

    def _add_param(self, label_text, default_val):
        row = len(self.params_frame.winfo_children()) // 2
        ttk.Label(self.params_frame, text=label_text).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(self.params_frame)
        entry.insert(0, default_val)
        entry.grid(row=row, column=1, sticky="w")
        setattr(self, f"{label_text}_entry", entry)

    def _get_param(self, name):
        return int(getattr(self, f"{name}_entry").get())

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="繼續訓練" if self.paused else "暫停訓練")

    def exit_program(self):
        self.interrupted = True
        self.paused = False
        self.root.quit()
        self.root.after(100, self.root.destroy)

    def start_training(self):
        self.interrupted = False
        n_particles = self._get_param("粒子數")
        n_generations = self._get_param("最大代數")
        rbf_hidden = self._get_param("RBF 隱藏層")

        self.trainer = PSOTrainer(n_particles=n_particles, n_generations=n_generations, rbf_hidden=rbf_hidden)
        self.best_particle = None

        for gen in range(n_generations):
            if self.interrupted:
                print("⚠️ 訓練被中斷")
                break
            self.root.update_idletasks()
            self.root.update()
            while self.paused:
                self.root.update()
            print(f"\n[GUI] Generation {gen}")
            for particle in self.trainer.particles:
                net = RBFNetwork(n_hidden=rbf_hidden)
                net.set_parameters(particle.position)
                _, success = self.trainer.simulate(net)
                if success:
                    self.best_particle = particle
                    self.eval_button.config(state=tk.NORMAL)
                    print("\U0001f3af 成功投射終點")
                    self.show_simulation(net, generation=gen, fitness=0.0)
                    return

            self.trainer.train_one_generation(gen)
            best_net = RBFNetwork(n_hidden=rbf_hidden)
            best_net.set_parameters(self.trainer.global_best_position)
            self.show_simulation(best_net, generation=gen, fitness=self.trainer.global_best_fitness)

        if not self.best_particle and self.trainer.global_best_position is not None:
            self.best_particle = self.trainer.global_best_position

        self.eval_button.config(state=tk.NORMAL)
        print("訓練結束")

    def start_evaluation(self):
        if not self.best_particle:
            print("未有成功結果")
            return

        net = RBFNetwork(n_hidden=self.trainer.rbf_hidden)
        net.set_parameters(self.best_particle.position if hasattr(self.best_particle, 'position') else self.best_particle)
        self.show_simulation(net)

    def show_simulation(self, net, generation=None, fitness=None):
        env = Playground()
        state = env.reset()
        step = 0
        max_steps = 300
        self.ax.clear()

        positions = [env.car.getPosition()]

        while not env.done and step < max_steps:
            angle = net.forward(state)
            env.car.setWheelAngle(angle)
            env.car.tick()
            env._checkDoneIntersects()
            state = env.state
            positions.append(env.car.getPosition())

            self.ax.clear()
            env.render(self.ax)
            xs = [p.x for p in positions]
            ys = [p.y for p in positions]
            self.ax.plot(xs, ys, 'r--', linewidth=1.5, alpha=0.6)
            self.canvas.draw()
            self.root.update()
            step += 1

        env.render(self.ax)
        xs = [p.x for p in positions]
        ys = [p.y for p in positions]
        self.ax.plot(xs, ys, 'r--', linewidth=1.5, alpha=0.6)
        self.canvas.draw()

        if generation is not None and fitness is not None:
            self.text_var.set(f"Generation: {generation} | Fitness: {fitness:.4f}")

    def reset_all(self):
        self.trainer = None
        self.best_particle = None
        self.ax.clear()
        self.text_var.set("Generation: 0 | Fitness: ---")
        self.canvas.draw()
        self.eval_button.config(state=tk.DISABLED)
        self.paused = False
        self.interrupted = True  # 中斷訓練循環
        self.pause_button.config(text="暫停訓練")
        print("🔄 重置完成")

if __name__ == "__main__":
    root = tk.Tk()
    app = PSOGUI(root)
    root.mainloop()
