from flask import Flask, render_template, request, url_for
from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.parser.pltlf import PLTLfParser
from pythomata.impl.symbolic import SymbolicDFA
from sympy import *
from itertools import combinations
import graphviz
import matplotlib.colors as colors
from ltlf2dfa.parser.ltlf import LTLfParser
import re
from pythomata import SymbolicAutomaton

import subprocess
import os
import datetime
import uuid
import base64


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
FUTURE_OPS = {"X", "F", "U", "G", "WX", "R"}
PAST_OPS = {"Y", "O", "S", "H"}

app = Flask(__name__)


def encode_svg(file):
    """Encode file to base64."""
    with open(file, "r") as image_file:
        encoded_string = base64.b64encode(image_file.read().encode("utf-8"))
    return encoded_string


def write_dot_file(dfa, name):
    """Write DOT file."""
    with open("{}/static/dot/{}.dot".format(PACKAGE_DIR, name), 'w') as fout:
        fout.write(str(dfa).replace(" size = \"7.5,10.5\";", "").replace("LR", "TB"))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/ltlf_syntax')
def ltlf_syntax():
    return render_template("ltlf_syntax.html")


@app.route('/pltlf_syntax')
def pltlf_syntax():
    return render_template("pltlf_syntax.html")




 ################################ ROBA NOSTRA ##############################################

def powerset(S):
    n = len(S)
    for r in range(0, n+1):
        for combo in combinations(S, r):
            yield set(combo)

def characteristic_function(S):
    cf = dict()
    for s in S:
        cf[str(s)]=True
    return cf 

def alphabet(dfa):
    alpha = set()
    for _, formula,_ in dfa.get_transitions():
        atoms = formula.atoms()
        for atom in atoms:
            alpha.add(str(atom))
    return alpha


def preC(dfa, controllable, uncontrollable, E):
    states = dfa.states
    pre_C = set()
    ps_c = powerset(controllable)
    for set_c in ps_c:
        for q in states.difference(E):
            pre_C.add(q)
            cf_c = characteristic_function(set_c) #it is an interpretation over st
            ps_u = powerset(uncontrollable)
            for set_u in ps_u:
                cf_u = characteristic_function(set_u)
                interpretation = dict(cf_c, **cf_u) 
                p = dfa.get_successor(q, interpretation)
                if p not in E:
                    pre_C.remove(q)
                    break     
    return pre_C
'''
def preC(dfa, controllable, uncontrollable, E):
    states = dfa.states
    pre_C = set()
    ps_c = powerset(controllable)
    for set_c in ps_c:
        for q in states.difference(E):
            pre_C.add(q)
            cf_c = characteristic_function(set_c) #it is an interpretation over st
            ps_u = powerset(uncontrollable)
            for set_u in ps_u:
                cf_u = characteristic_function(set_u)
                interpretation = dict(cf_c, **cf_u) 
                
                #guarda solo le transizioni che escono da q
                try:
                  trans_q = dfa._transition_function[q]
                  p = len(states)
                
                  for key in trans_q:
                    if trans_q[key].subs(interpretation) == True:
                      p = key
                      break
                
                  if p < len(states):
                    if p not in E:
                      pre_C.remove(q)
                      break    
                except:
                   print("nessuna transizione") 
    return pre_C
'''

def winning_dict(dfa, controllable, uncontrollable):
    #return a dict mapping states to when they were added to the winning set
    #if a state is not winning then it is not in the dict
    '''
    variables = alphabet(dfa)

    print("variables: ",variables)
    print("controllables: ",controllable)
    assert controllable.issubset(variables)  
    uncontrollable = variables.difference(controllable)
    '''
    winning_set = dfa.accepting_states
    winning_dict = dict()
    count = 0
    pre_C = winning_set
    while True:
        if len(pre_C)==0:
            break
        for s in pre_C:
            winning_dict[s]=count
        count+=1   
        print(count)  
        winning_set = winning_set.union(pre_C)
        pre_C = preC(dfa, controllable, uncontrollable, winning_set)
    return winning_dict

###################################################################coloriamo gli stati
def to_graphviz_winning_map(dfa, winningDict) -> graphviz.Digraph:
        """
        Convert to graphviz.Digraph object.

        :return: the graphviz.Digraph object.
        :raise ValueError: if it was not possible to compute the graph.
        """
        m = max(winningDict.values())
        if m != 0:       
          step = 1 / m
        else:
          step = 0
        #print("step: ", step)
        graph = graphviz.Digraph(format="svg")
        graph.node("fake", style="invisible")
        for state in dfa.states:
            if state in winningDict:
              c = ( step * winningDict[state] )
              color_node = colors.to_hex([ c, 1, 1-c])
            else:
              color_node = colors.to_hex([0.7 ,0.7 , 0.7])
            if state == dfa.initial_state:
                if state in dfa.accepting_states:
                    graph.node(str(state), root="true", shape="doublecircle", style ="filled", color =color_node)
                    
                else:
                    graph.node(str(state), root="true", style ="filled", color =color_node)
            elif state in dfa.accepting_states:
                graph.node(str(state), shape="doublecircle", style ="filled", color =color_node)
            else:
                graph.node(str(state), style ="filled", color =color_node)

        graph.edge("fake", str(dfa.initial_state))

        for (start, guard, end) in dfa.get_transitions():
            graph.edge(str(start), str(end), label=str(guard))

        return graph

def converter(expected):

    expected = str(expected)

    expected = expected.split("\n");
    init_states = []
    states = []
    transitions = []
    terminal_states = []
    for i in range(len(expected)):
        if ( ("->" in expected[i]) and ("init" not in expected[i])):
            temp = re.split("->", expected[i])
            temp[1] = temp[1].replace("[","")
            temp[1] = temp[1].replace("]", "")
            temp[1] = temp[1].replace(";", "")
            temp[1] = temp[1].replace("label=", "-")
            temp2 = temp[1].split("-")
            if(int(temp[0]) not in states):
                states.append(int(temp[0]))
            if (int(temp2[0]) not in states):
                states.append(int(temp2[0]))
            transitions.append( (int(temp[0]), str(temp2[1]), int(temp2[0])) )

    for i in range(len(expected)):
        if ( ("init" in expected[i])):
            for j in range(len(states)):
                state_string = str(states[j])+";"
                if(state_string in expected[i]):
                    init_states.append(states[j])
        if("doublecircle" in expected[i]):
            for j in range(len(states)):
                state_string = str(states[j])+";"
                if(state_string in expected[i]):
                    terminal_states.append(states[j])

    automaton = SymbolicAutomaton()
    #automaton = SymbolicDFA()
    state2idx = {}

    for i in range(len(states)):
        state_idx = automaton.create_state()
        state2idx[i] = state_idx
        if(states[i] in init_states):
            automaton.set_initial_state(state_idx)
        if(states[i] in terminal_states):
            automaton.set_accepting_state(state_idx, True)

    for i in range(len(transitions)):
        automaton.add_transition(
            transitions[i]
        )
    
    print("pre-determinize: ")
    determinized = automaton.determinize()
    print("pre-minimize: ")
    minimized = determinized.minimize()

    print("post-minimize: ")
    dfa = minimized
    #dfa = determinized

    #remove state 0
    #print(automaton.states)
    #automaton.remove_state(0)
    return dfa

def ltl2dfagame(ltl, controllables, uncontrollables, isLtlf):
      
      controllables = controllables.split(' ')
      uncontrollables = uncontrollables.split(' ')
      print("controllables :")
      print( controllables)

      print("uncontrollables :")
      print( uncontrollables)

      controllables = set(controllables)
      uncontrollables = set(uncontrollables)

      ### trasformazione in automata (LTL2DFA)
      
      if(isLtlf):
        parser = LTLfParser()
      else:
        parser = PLTLfParser()

      formula = parser(ltl)       

      print("converting to automaton....")
      dfa = formula.to_dfa()
      print(dfa)                          # prints the DFA in DOT format

      ### trasformazione in automata (PYTHOMATA)
      dfa = converter(dfa)

      ### stampo automa normale
      graph = dfa.to_graphviz()
      #graph.render(outputFileName+"_normal")

      ### calcolo winning dict
      print("\n\n#############  winning_dict ################")
      win_dict = winning_dict(dfa, controllables, uncontrollables)
      #print(win_dict)

      ### stampo winning map
      graph = to_graphviz_winning_map(dfa, win_dict)
      #graph.render(outputFileName)
      return graph



 ################################ FINE ROBA NOSTRA ########################################

@app.route('/dfa', methods=["POST"])
def dfa():
    formula_string = request.form["inputFormula"]
    control_set = request.form["controllables"]
    uncontrol_set = request.form["uncontrollables"]
    assert formula_string
    automa_name = "dfa_" + str(datetime.datetime.now()).replace(" ", "_") + "_" + str(uuid.uuid4())
    isLtlf = True
    if all(c in FUTURE_OPS for c in formula_string if c.isupper()):
        
        f_parser = LTLfParser()
        try:
            formula = f_parser(formula_string)
        except Exception as e:
            if request.form.get("exampleCheck1"):
                return render_template("dfa.html", error=str(e).encode("utf-8"))
            return render_template("index.html", error=str(e).encode("utf-8"))
    else:
        assert all(c in PAST_OPS for c in formula_string if c.isupper())
        isLtlf = False
        p_parser = PLTLfParser()
        try:
            formula = p_parser(formula_string)
        except Exception as e:
            if request.form.get("exampleCheck1"):
                return render_template("dfa.html", error=str(e).encode("utf-8"))
            return render_template("index.html", error=str(e).encode("utf-8"))

    dfa = ltl2dfagame(formula_string, control_set, uncontrol_set, isLtlf)
    write_dot_file(str(dfa), automa_name)
    subprocess.call('dot -Tsvg {} -o {}'.format("{}/static/dot/{}.dot".format(PACKAGE_DIR, automa_name),
                                                "{}/static/tmp/{}.svg".format(PACKAGE_DIR, automa_name)), shell=True)

    encoding = encode_svg("{}/static/tmp/{}.svg".format(PACKAGE_DIR, automa_name)).decode("utf-8")

    os.unlink("{}/static/dot/{}.dot".format(PACKAGE_DIR, automa_name))
    os.unlink("{}/static/tmp/{}.svg".format(PACKAGE_DIR, automa_name))

    return render_template("dfa.html",
                           formula=formula,
                           output=encoding)

if __name__== "__main__":
    app.run(debug=True)
