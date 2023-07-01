import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

def plot_section_1_1_results(file):
    Train_Steps = []
    Average_Per_Iteration_Reward = []
    Best_Mean_Reward_So_Far = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                Train_Steps.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Average_Per_Iteration_Reward.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Best_Mean_Reward_So_Far.append(v.simple_value)
    print(len(Train_Steps))
    print(len(Average_Per_Iteration_Reward))
    print(len(Best_Mean_Reward_So_Far))
    plt.plot(Train_Steps[:-2], Average_Per_Iteration_Reward[:-1], label = "Average Per Iteration Reward")
    plt.plot(Train_Steps[:-2], Best_Mean_Reward_So_Far, label = "Best Mean Reward So Far")
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Question 1: Basic Q-Learning Performance (DQN) for Ms. Pacman')
    plt.legend()
    plt.show()
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


def get_section_1_2_results(file):
  pass




if __name__ == '__main__':
    import glob
    logdir = 'data/q1_MsPacman-v0_13-10-2022_22-59-11/events*'
    eventfile = glob.glob(logdir)[0]
    
    Train_Steps = []
    Average_Per_Iteration_Reward = []
    Best_Mean_Reward_So_Far = []
    for e in tf.compat.v1.train.summary_iterator(eventfile):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                Train_Steps.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Average_Per_Iteration_Reward.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Best_Mean_Reward_So_Far.append(v.simple_value)
    print(len(Train_Steps))
    print(len(Average_Per_Iteration_Reward))
    print(len(Best_Mean_Reward_So_Far))
    plt.plot(Train_Steps[:-2], Average_Per_Iteration_Reward[:-1], label = "Average Per Iteration Reward")
    plt.plot(Train_Steps[:-2], Best_Mean_Reward_So_Far, label = "Best Mean Reward So Far")
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Question 1: Basic Q-Learning Performance (DQN) for Ms. Pacman')
    plt.legend()
    plt.show()


    # X, Y = get_section_results(eventfile)
    # for i, (x, y) in enumerate(zip(X, Y)):
    #     print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
    
