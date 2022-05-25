# MIT License
# Copyright (c) 2018 Valentyn N Sichkar (Valentyn N Sichkar. Reinforcement Learning Algorithms for global path planning // GitHub platform. DOI: 10.5281/zenodo.1317899)
# github.com/sichkar-valentyn
# For skeleton functions
#
# MIT License
# Copyright (c) 2021 Tunde Aderinwale, Daisuke Kihara, and Purdue University 
# For function implementations



# Importing libraries
import numpy as np  # To deal with data in form of matrices
import time  # Time is needed to slow down the agent and to see how he runs
from node import Node
from protein import Protein
import math
import pickle

model_states = []
# Creating class for the environment
class Environment:
    def __init__(self, chain_length,chain_list,protein_name,thresh,pair_thres,metro_thres,log,path,epsln,learning_rate,outpath,noiter,nip,ip,term_thres,unbound,ds,f_cla):
        self.clash_threshold = thresh
        self.pair_clash_threshold = pair_thres
        self.no_chains = chain_length
        self.protein_name = protein_name
        self.chain_list = chain_list
        self.model_status = []
        self.action_list = []
        self.non_interacting_pair = self.process_non_i_pair(nip)
        self.interacting_pair = self.process_int_pair(ip)
        self.build_interactions()
        self.built_chains = []
        #self.root_index = None
        #root_node = self.get_root()
        self.tree_root_node = Node('empty_root')
        self.logger = log
        self.path = path
        self.outpath = outpath
        self.lr = learning_rate
        self.eps = epsln
        self.metro_thres = metro_thres
        self.term_state_thres = term_thres
        self.unbound = unbound
        self.ds = ds
        self.fc = f_cla
        self.load_protein_class()
        self.eps_scalar = 1.0
        self.eps_decay = 1.0 / noiter
        self.repeat_track = 0
        self.old_action = 0



    def load_protein_class(self):
        prot = Protein(self.no_chains,self.chain_list,self.protein_name,self.clash_threshold,self.pair_clash_threshold,self.metro_thres,log=self.logger,path=self.path,lr=self.lr,eps=self.eps,outpath=self.outpath,term_thres=self.term_state_thres,dsize=self.ds,unbound=self.unbound,f_cla=self.fc)
        self.protein = prot

    def energy_stats(self):
        return self.protein.energydata

    #TODO: 
    #potential improvement is possible by appending and maintaining a single list instead of list of lists (not neccesarily true)
    def starting_position(self):
        self.model_status = []
        self.action_list = []
        #root_node = self.get_root()
        #self.model_status.append(root_node) # root of the complex/tree
        #self.build_model_chains()
        self.current_tree = self.tree_root_node

    def is_complex(self):
        return len(self.model_status) == self.no_chains - 1

    def is_forest(self):
        built_chains_len = len(list(set([item for sublist in self.model_status for item in sublist])))
        return (len(self.model_status) < self.no_chains - 1) and built_chains_len == self.no_chains

    def build_model_chains(self):
        self.built_chains = list(set([item for sublist in self.model_status for item in sublist]))
        

    def get_root(self):
        if self.root_index == None:
            self.root_index = np.random.choice([x for x in range(0,len(self.interaction))])
        return self.interaction[self.root_index]

    def process_non_i_pair(self,pair_args):
        nip_list = []
        if pair_args == '' or pair_args == None:
            return []
        else:
            pair_list = pair_args.split('-')
            for item in pair_list:
                R = self.chain_list.index(item[0])
                L = self.chain_list.index(item[1])
                nip_list.append([R,L])
                #nip_list.append([L,R])
        return nip_list


    def process_int_pair(self,pair_args):
        #A-BCD|B-A|D-ACE
        int_pair_data = {}
        if pair_args == '' or pair_args == None:
            return int_pair_data
        else:
            pair_list = pair_args.split('|')
            for item in pair_list:
                val = item.split('-')
                int_pair_data[self.chain_list.index(val[0])] = []
                for c in list(val[1]):
                    int_pair_data[self.chain_list.index(val[0])].append(self.chain_list.index(c))
        return int_pair_data

            


    def build_interactions(self):
        interaction_list = []
        for i in range(0,self.no_chains):
            for j in range(i,self.no_chains):
                if i==j:continue
                interaction_list.append([i,j])
        self.interaction = [x for x in interaction_list if x not in self.non_interacting_pair]

        print("Full Pair List " + str(interaction_list))
        print("None interacting Pair List " + str(self.non_interacting_pair))
        print("Interacting Pair List " + str(self.interacting_pair))
        print("Remaning interacting Pair List " + str(self.interaction))

    def get_next_state(self, current_state):
        pass


    def refresh(self):
        if self.protein.restart:
            self.refresh_tree()
            self.protein.restart = False
            return True
        else:
            return False

    def refresh_tree(self):
        self.tree_root_node = Node('empty_root')
        self.current_tree = self.tree_root_node

    # Function to reset the environment and start new Episode
    def reset(self):
        #TODO self.state = [] self.nodes = []
        return self.starting_position()

    # Function to get the next observation and reward by doing next step
    # Next state selection should be base on tree policy... use uct for state transition
    def step(self, action,episode):
        # Current state of the agent
        old_action = 0
        state = self.built_chains # use a list to keep track of built subunits 
        #print("in step function, current model status: ",self.model_status)
        # ignoring the first action since we are moving from empty state. actions from there is not important
        if len(self.model_status) == 0:
            pass
        else:
            #regular actions selection
            if type(action) is np.int64:
                self.action_list.append(action)
                old_action = action
            else:
                #happens when there is a repeat. top 5 actions are returned
                self.action_list.append(action[(self.repeat_track-1)])
                old_action = action[(self.repeat_track-1)]
        #print('Current action_list : ',self.action_list)

        repeat = False
        

        # Calculating the reward for the agent
        if self.is_complex():
            # print('NOW IN GOAL STATE')
            # print("Current model status: ",self.model_status)
            # print('Current action_list : ',self.action_list)            

            const_status = False #self.protein.validate_progress_with_contrainst()

            if const_status:
                reward = -20
                done = True
                next_state = 'goal'
                self.protein.log_decoy_info(self.model_status,self.action_list,episode)
                self.backup_tree_value(reward)
            else:
                reward = self.get_model_score(episode)
                done = True
                next_state = 'goal'
                self.backup_tree_value(reward)
        elif self.is_forest():
            reward = -5
            done = True
            next_state = 'forest'
            print('FOREST DETECTED... penalty reward')
            print(self.model_status,self.action_list)
            self.backup_tree_value(reward)
        else:
            reward = 0
            done = False


            # validat progress before moving
            clash_check = self.protein.validate_progress(self.model_status,self.action_list)
            #clash_check = False
            if clash_check and (self.repeat_track <5):
                #reject current action
                self.repeat_track += 1
                self.old_action = self.action_list.pop()
                #stay in same position
                next_state = self.model_status
                repeat = True

            else:
                self.repeat_track = 0
                # Moving the agent according to the action
                best_child = self.select_best_child()

                if best_child == 'NOCHILD':
                    reward = -5
                    done = True
                    next_state = 'forest'
                    print('FOREST DETECTED in build... penalty reward')
                    print(self.model_status,self.action_list)
                    self.backup_tree_value(reward)
                else:
                    self.model_status.append(best_child.name)
                    self.build_model_chains() #adds new child to the complex
                    self.move_tree(best_child)

                    # Updating next state
                    next_state = self.model_status

        return next_state, reward, done, repeat , old_action
    def final(self):
        model_states = self.model_status


        #Dump current model status to disk
        with open(self.outpath + self.protein_name + '_' + str(self.lr) +'_model_status', 'wb') as fp:
            pickle.dump(model_states, fp)

        #Dump current tree to disk
        with open(self.outpath + self.protein_name + '_' + str(self.lr) +'_model_tree','wb') as fo:
            pickle.dump(self.tree_root_node,fo)
        
        
        #TODO print Tree topology
        node = self.tree_root_node#self.current_tree

        #now print path to 
        path_string =  ''

        while len(node.children) != 0:
            bestscore= float("-inf")
            best_child = []
            for c in node.children:
                exploit=c.reward/c.visits
                if exploit >= bestscore:
                    best_child.append(c)
                    bestscore=exploit

            bc = np.random.choice(best_child)

            path_string += str(bc.name) + '-->' 
            node = bc

        return path_string



    def load_tree_for_continue(self):
        #Load model status
        with open (self.outpath + self.protein_name + '_' + str(self.lr) +'_model_status', 'rb') as fp:
            self.model_status = pickle.load(fp)

        #Load current tree
        with open (self.outpath + self.protein_name + '_' + str(self.lr) +'_model_tree', 'rb') as fp:
            self.current_tree = pickle.load(fp)


    
    def epsdecay(self):
        if self.eps_scalar > 0:
            self.eps_scalar -= self.eps_decay

    def backup_tree_value(self,r):
        node = self.current_tree
        while node!=None:
            node.visits+=1
            node.reward+=r
            node=node.parent

    def move_tree(self,best_child):
        self.current_tree = best_child

    def will_not_cause_forest(self,parent,child):
        if parent == 'empty_root':
            return True
        if child[0] in list(parent) or child[1] in list(parent):
            return True
        
        return False
        # print('parent :', list(parent)[0])
        # print('child :', child[0])
        # return True

    def child_is_interacting_with_parent(self,parent,child):
        if parent == 'empty_root': # empty root everything is interacting
            return True
        if not self.interacting_pair: # no interacting pair information is parsed so return true
            return True
        parent_list = list(parent)
        if parent_list[0] in self.interacting_pair:
            if child[0] in self.interacting_pair[parent_list[0]] or child[1] in self.interacting_pair[parent_list[0]]:
                if str(child)  == str([parent_list[0],child[0]]) or str(child) == str([child[0],parent_list[0]]):
                    return True
                if str(child)  == str([parent_list[0],child[1]]) or str(child) == str([child[1],parent_list[0]]):
                    return True
        if parent_list[1] in self.interacting_pair:
            if child[0] in self.interacting_pair[parent_list[1]] or child[1] in self.interacting_pair[parent_list[1]]:
                if str(child)  == str([parent_list[1],child[0]]) or str(child) == str([child[0],parent_list[1]]):    
                    return True
                if str(child)  == str([parent_list[1],child[1]]) or str(child) == str([child[1],parent_list[1]]):
                    return True
        return False

    def expand_tree(self):
        list_of_possible_child = self.build_children()
        for x in list_of_possible_child:
            if x not in self.current_tree.children and self.will_not_cause_forest(self.current_tree.name,x) and self.child_is_interacting_with_parent(self.current_tree.name,x):
                self.current_tree.add_child(x)

    def expand_tree_with_forest(self):
        list_of_possible_child = self.build_children()
        for x in list_of_possible_child:
            if x not in self.current_tree.children:# and self.will_not_cause_forest(self.current_tree.name,x):
                    self.current_tree.add_child(x)

    # this should be changed to UCT bounds for child selection
    def select_best_child(self):
        scalar = self.eps_scalar #0.5/math.sqrt(2.0) #TODO change this to a parameter and should be set
        if len(self.current_tree.children) == 0: 
            self.expand_tree_with_forest()
            if len(self.current_tree.children) == 0:  # if there is still no children, lets allow poteitial for forest
                self.expand_tree_with_forest()
                if len(self.current_tree.children) == 0:
                    return 'NOCHILD'
            return np.random.choice(self.current_tree.children) # randomly select one child since they all have same value
            
        #introduce random selection of child similar to qtable approach
        if np.random.uniform() < self.eps_scalar:
            return np.random.choice(self.current_tree.children)
        else:
            bestscore= float("-inf")
            bestchildren=[]
            c_score_list = {} #debugging purpose
            for c in self.current_tree.children:
                exploit=c.reward/c.visits
                explore=math.sqrt(2.0*math.log(self.current_tree.visits)/float(c.visits))    
                score=exploit+scalar*explore
                c_score_list[str(c.name)] = score
                if score==bestscore:
                    bestchildren.append(c)
                if score>bestscore:
                    bestchildren=[c]
                    bestscore=score
            if len(bestchildren)==0:
                # print("OOPS: no best child found, probably fatal")
                # Returning random child
                bc = np.random.choice(self.current_tree.children)

      
        bc = np.random.choice(bestchildren)
        # print(bc)
        #print('Best child to return is : ',bc.name, ' with value : ', bc.reward)
        return bc

    def build_children(self):
        current_state = self.model_status
        first_set = set(map(tuple, self.interaction))
        second_set = set(map(tuple, self.model_status))
        remaining_child = first_set.symmetric_difference(second_set)
        remaining_pairs = [list(x) for x in remaining_child] # convert from set back to list

        #filter out redundant pair
        good_pairs = []
        for x in remaining_pairs:
            built_pairs = []
            built_pairs = self.model_status.copy()
            built_pairs.append(x)
            #check to ensure adding this new pair to the current topology will increase the chains builts
            if len(list(set([item for sublist in built_pairs for item in sublist]))) > len(list(set([item for sublist in self.model_status for item in sublist]))):
                good_pairs.append(x)
        return good_pairs

    def get_model_score(self,episode):
        return self.protein.make_and_score_complex(self.model_status,self.action_list,episode)
        #return self.protein.make_and_score_complex_new(self.model_status,self.action_list,episode)
        #return np.random.choice([x for x in range(1,20)])

    def print_energy_stats(self):
        return self.protein.energy_tragetory
    def print_metro_stats(self):
        self.protein.print_metro_stats()


def final_states():
    return model_states
