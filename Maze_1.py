# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:13:46 2020

@author: ryukei
"""


import time
import numpy as np
import tkinter as tk

class Maze_1(tk.Toplevel,object):
    """The class of Maze,subclass of tkinter.Toplevel
       information:generate a maze that the agent is going to explorate.
                   The maze can generate the reward belongs to the action that took by the agent.
       Attributes:
           maze_size : the size of maze by describe a set of (width,height). eg.(17,17)
           trap_number : the number of traps in the maze,the default is 1 .
       
       
    
    """
    
    def __init__(self,maze_size = (7,7),trap_number=0):
        """initiate the maze."""
        
        super(Maze_1,self).__init__()
        
        # A:robot 1 C:goal T:trap 0:wall 1:road
        self.maze = [['A',1,1,1,1,0,0],
                     [0,1,1,1,1,0,1],
                     [0,1,1,0,'C1',0,1],
                     [1,1,0,0,0,1,1],
                     [1,0,'C2',0,1,1,0],
                     [1,0,1,1,1,1,0],
                     [0,0,1,1,1,1,'B']]
        
        
        self.img_robot_1 = tk.PhotoImage(file='image/agent1.ppm') # the image of robot
        self.img_robot_2 = tk.PhotoImage(file='image/agent2.ppm') # the image of robot
        self.img_treasure = tk.PhotoImage(file='image/treasure_new.ppm') # the image of treasure
        self.img_bomb = tk.PhotoImage(file='image/bomb_new.ppm') # the image of bomb
        self.img_up = tk.PhotoImage(file='image/up.ppm') # the image of up
        self.img_down = tk.PhotoImage(file='image/down.ppm') # the image of down
        self.img_left = tk.PhotoImage(file='image/left.ppm') # the image of left
        self.img_right = tk.PhotoImage(file='image/right.ppm') # the image of right
        
        self.WIDTH = len(self.maze[0]) # the maze's width
        #print('Width of maze is {}.'.format(self.WIDTH))
        self.HEIGHT = len(self.maze) # the maze's height
        #print('Height of maze is {}.'.format(self.HEIGHT))
        
        self.UNIT = 40 # the pixel of one area of the maze #おそらく一マス40ピクセル
        
        self.trap_number = trap_number # the number of traps in the maze        
        self.title('MAZE') # the title of the maze
        self.geometry('{0}x{1}'.format(self.WIDTH*self.UNIT,self.HEIGHT*self.UNIT)) # create a window of tkinter        
        self.actions = ['up','down','left','right'] # the actions which can be chosen by agent in the maze
        self.actions_num = len(self.actions) # get the number of actions 
        self.observation_space = [4,4]
        self.build_maze() # initiate the maze
        
        
    def build_maze(self):
        """initiate the maze
           what the maze looks like is setted in this method.
        

        Returns
        -------
        None.

        """
        # create the canvas: need the color of background is white,and width and height of the maze
        self.canvas = tk.Canvas(self,bg='white',width=self.WIDTH*self.UNIT,height=self.HEIGHT*self.UNIT)
 
        # get all the coordinate of x and y
        x = [] # eg. [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        y = [] # eg. [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        x.append(0)
        y.append(0)
               
        for column in range(1,self.WIDTH+1):
            new_x = x[0] + column * self.UNIT
            x.append(new_x)
            
        #print("x is {}".format(x))
        
        for row in range(1,self.HEIGHT+1):
            new_y = y[0] + row * self.UNIT
            y.append(new_y)
        
        #print("y is {}".format(y))
        
        # create the edge of the maze       
        self.canvas.create_line(x[0],y[0],x[0],y[self.HEIGHT])
        self.canvas.create_line(x[0],y[self.HEIGHT],x[self.WIDTH],y[self.HEIGHT])
        self.canvas.create_line(x[self.WIDTH],y[self.HEIGHT],x[self.WIDTH],y[0])
        self.canvas.create_line(x[self.WIDTH],y[0],x[0],y[0])
        
        # set the coordinate of center point 
        origin = np.array([self.UNIT/2,self.UNIT/2]) # eg.[20,20]
        bomb_id = 1
        goal_id = 0
        self.walls = [] # insert the coordinate of all walls 
        self.goals = []
        self.bombs = []
        self.robot_1_origin_center = []
        self.robot_2_origin_center = []
        self.goal_center_1 = []
        self.goal_center_2 = []
        # draw the maze
        for row_id in range(self.HEIGHT):
            for column_id in range(self.WIDTH):
                
                center_coordinate = origin + np.array([column_id * self.UNIT,row_id * self.UNIT])
                if self.maze[row_id][column_id] == 'A' : # robot 1 
                # set the coordinate of the agent's center point
                    self.robot_1_origin_center = center_coordinate
                    # create the robot 1 in the canvas use the image of robot
                    #canvas.create_imageはアイテムidを返す??  #self.robot_1はおそらくアイテムID
                    self.robot_1 = self.canvas.create_image(center_coordinate[0],center_coordinate[1],image=self.img_robot_1)
                    
                elif self.maze[row_id][column_id] == 'B' : # robot 2
                    # create the robot 1 in the canvas use the image of robot
                    self.robot_2_origin_center = center_coordinate
                    self.robot_2 = self.canvas.create_image(center_coordinate[0],center_coordinate[1],image=self.img_robot_2)
                
                elif self.maze[row_id][column_id] == 'T' : # trap
                    self.bombs.append(self.canvas.coords(self.canvas.create_image(center_coordinate[0],center_coordinate[1],image=self.img_bomb)))
                    bomb_id += 1
                elif self.maze[row_id][column_id] == 'C1'  : # treasure 1
                    self.goal_1 = self.canvas.create_image(center_coordinate[0],center_coordinate[1],image=self.img_treasure)
                    self.goal_center_1 = center_coordinate
                    self.goals.append(self.canvas.coords(self.goal_1))
                    goal_id += 1
                
                elif self.maze[row_id][column_id] == 'C2'  : # treasure 2
                    self.goal_2 = self.canvas.create_image(center_coordinate[0],center_coordinate[1],image=self.img_treasure)
                    self.goal_center_2 = center_coordinate
                    self.goals.append(self.canvas.coords(self.goal_2))
                    goal_id += 1
               
                elif self.maze[row_id][column_id] ==  0  : # wall
                    # creat all the wall in the canvas
                    self.canvas.create_rectangle(x[column_id],y[row_id],x[column_id+1],y[row_id+1],fill='gray',outline='')
                    self.walls.append(self.get_coordinate(x[column_id],y[row_id],x[column_id+1],y[row_id+1]))
                
        
        
        
        #print("bombs is {}".format(self.bombs))#[[660.0, 420.0], [100.0, 580.0], [580.0, 660.0]]
        #print("treasure is {}".format(self.goals))#[[100.0, 180.0], [660.0, 660.0]]
        #print("wall is {}".format(self.walls))
        
        
        
        
        
        
        #print("Start:agent's coordinate is "+str(self.sense_agent()))
        
        
        # robots next state
        self.robots_next_state = {}
        
        
        self.goal_1_sign = True# 
        self.goal_2_sign = True#
        self.goal_sign = self.goals
        #self.goals.append(self.canvas.coords(self.goal_1))
        #self.goals.append(self.canvas.coords(self.goal_2))
       
        self.goal_counter = goal_id
        
        
        self.done = False
        self.episode_done = False
        self.result = 0
        #self.draw_goals()
        
        
        
        
        #self.update()
        #self.render()
       
        self.canvas.pack()

    
    def get_env_info(self):
        env_info = {'state_shape':4,'n_actions':4,'n_agents':8}
        return env_info
        
    
    def reset(self):
        """reset the maze
           If the agent is going to the goal or trap , or the episode is done,
           the maze will be reset
        

        Returns
        -------
        set
            the coordinate of the agent in the maze.

        """
        #print("reset the environment!")
        self.update()
        time.sleep(0.5)
        
                
        
        self.canvas.delete(self.robot_1) # delete the agent in the maze
        self.canvas.delete(self.robot_2)
       
        
        # create the agent in the canvas use the image of robot
        self.robot_1 = self.canvas.create_image(self.robot_1_origin_center[0],self.robot_1_origin_center[1],image=self.img_robot_1)
        #print("Start:agent's coordinate is "+str(self.sense_agent()))
        self.robot_2 = self.canvas.create_image(self.robot_2_origin_center[0],self.robot_2_origin_center[1],image=self.img_robot_2)
        
        self.robots_next_state = {}
        self.robots_next_state['robot_1'] = (0,0)
        self.robots_next_state['robot_2'] = (6,6)
        
        self.draw_goals()
        self.result = 0
        #return self.canvas.coords(self.robot)
        return
    
    def draw_goals(self):
        if self.goal_1_sign == False:
            self.goal_1 = self.canvas.create_image(self.goal_center_1[0],self.goal_center_1[1],image=self.img_treasure)
            self.goal_1_sign = True
        if self.goal_2_sign == False:
            self.goal_2 = self.canvas.create_image(self.goal_center_2[0],self.goal_center_2[1],image=self.img_treasure)
            self.goal_2_sign = True
        self.goals = []
        self.goals.append(self.canvas.coords(self.goal_1))
        self.goals.append(self.canvas.coords(self.goal_2))
        self.goal_counter = 2
        
        # print("self.goal_1 = ",self.goal_1)
        # print("self.goals = ",self.goals)
        #print("goals is {}".format(self.goals))
        #print('the number of goals is {}'.format(self.goal_counter))
        return
    
    def step(self,joint_actions):
        """move the agents in the maze
           The agents will move to next point by their actions.
           robot1:action_1
           robot2:action_2

        Parameters
        ----------
        action_1 : string
                 eg . actions = ['up','down','left','right']
        action_2 : string
                 eg . actions = ['up','down','left','right']
        Returns
        -------
        next_state: set
                    the next coordinate of the agent in the maze.
        reward: float number
                the reward belongs to the maze.
        done: boolean
            the episode is end or not. if is True,means episode is end ,else False means not end.

        """
        # print("====================Maze_1.step===============================")
        # print("joint_actions = ",joint_actions)
        robot_list = [('1',self.robot_1),('2',self.robot_2)]
        robots =  dict(robot_list)
        # print("self.robot_1 = ",self.robot_1)
        # print("robot_list = ",robot_list)
        # print("robots = ",robots)
        #actions_list = [('robot_1',joint_actions[0]),('robot_2',joint_actions[1])]
        #actions = dict(actions_list)
        
        names = ['robot_1','robot_2']
        
        
        rewards = 0 # robots total rewards
        
        # check the robot is go to the trap or goal
        robots_done = []
        self.episode_done = False

    
        
        for robot in range(2) :
            #print(robot)
            
            # print("-----------robot",(robot+1),"-----------")
            current_state = self.canvas.coords(robots[str(robot+1)]) # get the current coordinate of the agent 1. eg.[20,100]
            # print("current_state = ",current_state)
            cordin = self.get_next_state_coordinate(current_state[0],current_state[1])#cordinはおそらく左上を(0,0)とした現在のマスの座標を表している
            # print("cordin = ",cordin)
            #print('The current state of robot '+str(robot + 1)+' is '+str(cordin))
            base_action = np.array([0,0]) # this will be use to calculate the next coordinate of the agent.
            action = joint_actions[robot] # get the actions of robots
            # print("action = ",action)
            # check the action
            # in this part , MDP the transition probability can be done.Now , transition probabiliy equal to 1.
            if action == 'up': # up
                    base_action[1] -= self.UNIT # eg.[0,-40]
            elif action == 'down': # down
                    base_action[1] += self.UNIT # eg.[0,40]
            elif action == 'left': # left
                    base_action[0] -= self.UNIT # eg.[-40,0]
            else : # right
                    base_action[0] += self.UNIT # eg.[40,0]
            # print("base_action",base_action)
            # guess if the agent take that action , what the next state will be.
            guess_next_state = []
            guess_next_state.append(current_state[0]+base_action[0]) # get the next_state:x xの座標がピクセル単位で計算される
            guess_next_state.append(current_state[1]+base_action[1]) # get the next_state:y yの座標がピクセル単位で計算される
            # get the coordinate of next state if the agent do take that action
            
            #guess_next_stateにはエージェントの遷移後の座標が格納されている
            # print("guess_next_state",guess_next_state)
            guess_next_coordinate = self.get_next_state_coordinate(guess_next_state[0],guess_next_state[1]) #guess_next_coordinateはおそらく左上を(0,0)とした遷移後のマスの座標を表している
            # print("guess_next_coordinate",guess_next_coordinate)
            #print("Guess ! the next state of agent may be in "+str(guess_next_coordinate))
    
            #遷移後の座標をもとに，迷路の外に出てしまったかどうかを判定
            # check the agent is going out of edge or not.
            out_of_edge = False
            if guess_next_state[0] < 0 :   
                # go out of the left edge
                base_action[0] = 0
                base_action[1] = 0
                out_of_edge = True
                
            elif guess_next_state[0] > self.WIDTH*self.UNIT:
                # go out of the right edge
                base_action[0] = 0
                base_action[1] = 0
                out_of_edge = True
                  
            
            if guess_next_state[1] < 0 :
                # go out of the up edge
                base_action[0] = 0
                base_action[1] = 0
                out_of_edge = True
                
            elif guess_next_state[1] > self.HEIGHT*self.UNIT:
                # go out of the down edge
                base_action[0] = 0
                base_action[1] = 0
                out_of_edge = True                    
                          
            # check the agent is going to the wall or not
            is_strike_wall = False
            # print("self.walls = ",self.walls)
            for item in self.walls:
                if guess_next_coordinate == item:
                    is_strike_wall = True
                    base_action[0] = 0
                    base_action[1] = 0
                    
            # check whether the other robots is in the next grid 
            #guess_next_state　,guess_next_coordinate 両方をチェックした結果，迷路から出ていなければ
            is_strike_robot = False
            if out_of_edge == False and is_strike_wall == False :
                # print("self.robots_next_state.values() = ",self.robots_next_state.values())
                for robot_state in self.robots_next_state.values():
                    if guess_next_coordinate == robot_state:
                        is_strike_robot = True
                
                                   
            # move the agent to the next coordinate #グラフ上で，遷移後の座標にエージェントを移動させる
            self.canvas.move(robots[str(robot+1)],base_action[0],base_action[1])
            # get the next state of the agent in the maze
            next_state = self.canvas.coords(robots[str(robot+1)]) # eg . [20,100]
            
            # print("next_state = ",next_state)
            next_state_coordinate = self.get_next_state_coordinate(next_state[0],next_state[1]) # eg . (0,2)
            # print("next_state_coordinate = ",next_state_coordinate)
            #print("The next state of agent will be in "+str(next_state_coordinate))
            
            #robot iに関して遷移後の状態を記入
            self.robots_next_state[names[robot]] = next_state_coordinate
            
            reward = self._get_reward(next_state,out_of_edge,is_strike_wall,is_strike_robot)
            #print('The reward of this step is {}'.format(reward))
            rewards = rewards + reward
            
            robots_done.append(self.done)
            
            
        # check the done
        #print(robots_done)
        for item in robots_done:
            if item == True:
                self.episode_done = True#二つの宝を手に入れたとき，罠にかかった時にTrueとなる

        #print('The next state of robots in the maze is {}'.format(self.robots_next_state))
        observations = self._get_observations(self.robots_next_state)
            
        
        # print("self.result = ",self.result)
        return observations,rewards,self.episode_done,self.result
    
    def _get_observations(self,robots_next_state):
        
        robot1_state = robots_next_state['robot_1']
        #print('Next state of robot 1 is {}'.format(robot1_state))
        robot2_state = robots_next_state['robot_2']
        #print('Next state of robot 2 is {}'.format(robot2_state))
        robot1_obs = []
        robot2_obs = []
        all_obs = []
        
        if self.episode_done == True:
            robot1_next_state = None
            robot2_next_state = None
            all_obs.append(np.array([0,0,0,1]))#大事なところ
            all_obs.append(np.array([0,0,1,0]))#大事なところ
            
        else:
            robot1_next_obs = self._cal_observations(robot1_state)
            for id in robot1_next_obs.keys():
                robot1_obs.append(robot1_next_obs[id])
            #print('Next obs of robot1 is {}'.format(robot1_obs))
            robot2_next_obs = self._cal_observations(robot2_state)
            for id in robot2_next_obs.keys():
                robot2_obs.append(robot2_next_obs[id])
            #print('Next obs of robot2 is {}'.format(robot2_obs))
            all_obs.append(np.array(robot1_obs))
            all_obs.append(np.array(robot2_obs))
        
        return all_obs
            
    def sense_observations(self,robot_id):
        
        robots_coord = {}
        
        robot1_state = self.canvas.coords(self.robot_1)
        robot2_state = self.canvas.coords(self.robot_2)
        
        robot_1_coord = self.get_next_state_coordinate(robot1_state[0],robot1_state[1]) 
        robot_2_coord = self.get_next_state_coordinate(robot2_state[0],robot2_state[1]) 
        
        robots_coord['robot_1'] = robot_1_coord
        robots_coord['robot_2'] = robot_2_coord
        robots_obs = self._get_observations(robots_coord)
        
        if self.done == True:
            if robot_id == '1' :
                return np.array([0,0,0,1])#大事なところ
            else:
                return np.array([0,0,1,0])#大事なところ
        else:    
            
            if robot_id == '1' :
                return np.array(robots_obs[0])
            else:
                return np.array(robots_obs[1])
            
     
    def _cal_observations(self,next_state):
        
        x = next_state[0]
        #print(x)
        y = next_state[1]
        #print(y)
        
        up_obs = (x,y-1)
        down_obs = (x,y+1)
        left_obs = (x-1,y)
        right_obs = (x+1,y)
        
        obs = {0:up_obs,1:down_obs,2:left_obs,3:right_obs}
        #print(obs)
        obs_sign = {}
        # wall and edge : 0 ; road : 1 ; trap : 3 ; treasure : 2 ; robot : 4
        
        
        for id in range(4):
            item = obs[id]
            #print(item)
            is_road = True
            for bomb in self.bombs:
                if item == self.get_next_state_coordinate(bomb[0],bomb[1]):
                    obs_sign[id] = 3
                    is_road = False
                    #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            for goal in self.goal_sign:
                if item == self.get_next_state_coordinate(goal[0],goal[1]):
                    obs_sign[id] = 2
                    is_road = False
                    #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            
                
            if item[0] < 0 or item[0] > (self.WIDTH - 1) or item[1] < 0 or item[1] > (self.HEIGHT - 1):
                obs_sign[id] = 0
                is_road = False
               
            
            for wall in self.walls:
                if item == wall:
                    obs_sign[id] = 0
                    is_road = False 
                   
                    
           
            for robot_state in self.robots_next_state.values():
                if item == robot_state:
                    obs_sign[id] = 4
                    is_road = False
                    #print('####################################################################################################')
           
            if is_road == True:    
                obs_sign[id] = 1      
            
                    
        #print('obs is {}'.format(obs_sign))#{up:,down:,left:,right:}
        
        return obs_sign
            
           
            
        

    
    def _get_reward(self,next_state,out_of_edge,is_strike_wall,is_strike_robot):  
        # check the agent is going to the goals or not
            # print("self.goals = ",self.goals)
            for item in self.goals:
                if next_state == item: #ゴールした時
                    reward = 10 # goal:+10
                    
                    #rewards = rewards + reward
                    # print("self.goal_counter = ",self.goal_counter)
                    self.goal_counter -= 1
                    # print("self.goal_counter = ",self.goal_counter)
                    if self.goal_counter == 1 :
                        self.done = False
                        self.result = 1
                        # print('This is the first goal.congratulations!')
                        
                    if self.goal_counter == 0 :
                        self.done = True
                        self.result = 2
                        reward = 20 # goal:+20
                        # print('This is the second goal.congratulations!')   
                    
                    self.goals.remove(item)
                    #print(self.goals)
                    #print(self.goal_counter)
                    if next_state == self.canvas.coords(self.goal_1) :
                        
                        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$----Treasure----$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        
                        self.canvas.delete(self.goal_1)
                        self.goal_1_sign = False
                        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                        self.update()
                        time.sleep(0.5)
                        
                        
                        #return next_state_coordinate,reward,done
                    else:
                        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$----Treasure----$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        
                        self.canvas.delete(self.goal_2)
                        self.goal_2_sign = False
                        #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                        self.update()
                        time.sleep(0.5) 
                        next_state_coordinate = 'terminal' # go back to the start point 
                        #return next_state_coordinate,reward,done
                    return reward 
                        
            
            
            # set the reward belongs to different situations
            if out_of_edge == True:
                reward = -0.1 # edge:-0.5
                #rewards = rewards + reward
                self.done = False
                #print('This is the edge. Can not go out of the maze !')
                return reward
                #return next_state_coordinate,reward,done
            
            if is_strike_wall == True:
                reward = -0.1 # wall:-0.5
                #rewards = rewards + reward
                self.done = False
                #print('This is the wall.Can not go into the wall !')
                return reward
                #return next_state_coordinate,reward,done
            
            # check the strack
            if is_strike_robot == True:
                reward = -1
                #rewards = rewards + reward
                self.done = False
                #print('Strike with the other robot!')
                return reward
                #return next_state_coordinate,reward,done
            
            
            for bomb in self.bombs :
                if next_state == bomb :
                    reward = -1 # trap:-10
                    self.done = True
                    next_state_coordinate = 'trap' # go back to the start point
                    #print('This is a trap. Please restart in the start point.')
                    return reward
            
            
            reward = -0.04 # road:-0.04
            
            #rewards = rewards + reward
            self.done = False
            #print('This is the road.')
            return reward
        
           
     
    def render(self):
        """update the maze
           update the maze in 0.1s,so the moving of agent can be seen.
        

        Returns
        -------
        None.

        """
        time.sleep(0.01) # use 0.1s to update the maze
        self.update()                                                                
        
   
    def sense_agent(self,robot_num):
        """sense the state of the agent in the maze.
           use the method to get the state of the agent in the maze.
        

        Returns
        -------
        coordinate : set
                     the state of the agent in the maze. eg . (0,2)

        """
        robots_list = {('1',self.robot_1),('2',self.robot_2)}
        
        state = self.canvas.coords(robots_list[robot_num]) # get the coordinate of agent's center point 
        x_new = int(state[0]/self.UNIT)
        y_new = int(state[1]/self.UNIT)
        coordinate = (x_new,y_new)       
        return coordinate
   
    def sense_coordinate(self,robot_num):
        """sense the coordinate of the agent in the maze.
           use the method to get the coordinate of the agent in the maze.
        

        Returns
        -------
        coordinate : list
                     the coordinate of the agent in the maze. eg . [20,100]

        """
        robots_list = {('1',self.robot_1),('2',self.robot_2)}
        coordinate = self.canvas.coords(robots_list[robot_num])
        return coordinate

    def get_next_state_coordinate(self,next_state_x,next_state_y):
        """get the coordinate of next state
           use this method to guess the next coordinate of the agent if take some action.
        

        Parameters
        ----------
        next_state_x : integer
                       the next_state:x
                         
        next_state_y : integer
                       the next_state:y
        Returns
        -------
        coordinate : set
                     the state of the agent in the maze. eg . (0,2)

        """
        x_new = min(int(next_state_x/self.UNIT),self.WIDTH-1)
        y_new = min(int(next_state_y/self.UNIT),self.HEIGHT-1)
        coordinate = (x_new,y_new)        
        return coordinate
        
    
    def get_coordinate(self,x0,y0,x1,y1):
        """get the coordinate in the maze 
           use this method to get the coordinate of walls in the maze
        

        Parameters
        ----------
        x0 : integer
             the first point:x
        y0 : integer
             the first point:y
        x1 : integer
             the second point:x
        y1 : integer
             the second point:y

        Returns
        -------
        coordinate : set
                     the state of the agent in the maze. eg . (1,2)

        """
        x_new = int(x0/self.UNIT)
        y_new = int(y0/self.UNIT)
        coordinate = (x_new,y_new)      
        return coordinate
        
    def show_route(self,coordinate,action):
        """show the route 
           use this method to show the route in the maze
        
        Parameters
        ----------
        coordinate : list
                     the coordinate of agent in the maze
        action : string
                 the action of the agent

        Returns
        -------
        None.

        """
        
        #print("coordinate of agent is :"+str(coordinate))
        
        
        
        
        if action == 'up':
            img_direction = self.img_up
        elif action == 'down':
            img_direction = self.img_down
        elif action == 'left':
            img_direction = self.img_left
        else:
            img_direction = self.img_right
            
        self.canvas.create_image(coordinate[0],coordinate[1],image=img_direction)
        self.render()
    
        
    