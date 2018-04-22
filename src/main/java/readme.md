#Artificial Intelligent (Projekt)

**Author**: _Björn Frohberg_, Software-Developer, Germany, bf2782@web.de

## Introduction

### Goals

This project shows how a neural network work and how to use it, will be
descripted in this documentation.

&check; Create neural network library projekt <br/>
&check; Create the project for JAVA 1.8 <br/>
&check; Begin documentation <br/>
* Create a neural network (nn) type
* Create a nn layer type
* Create a nn layer neuron
* Create a nn layer neuron binding
* Make the neural network saveable to be restorable
* Make the neural network restorable from load
* Create a test to do a smoke testing on the nn

### Mechanics

**Task** Neural Network (nn) : Type

######_Business Value_
We need a handler to do operations representing the neural network.

######_Requirements_
This are operations the nn must have
* prepare the neural network with layers and their neurons
* test given inputs and return predicted output values therefore
* train the neural network, by using test with a learning gradient <br/>
* implement a routine to use Training Session plus amount of iterations (generations) to use train operations on each Training unit
* make the training based on a Training Session
* make a Training Session contain Training Units
* make a Training Unit contain input values and desired / expected output values

<pre><code></code></pre>

**Task** Training Unit : Type

######_Business Value_
For a training session I need a Training Unit (type) to enlist training definitions.

######_Requirements_
* input values : array of floating numbers
* desired / expected output values : array of floating numbers

<pre><code></code></pre>

**Task** Training Session : Type

######_Business Value_
To be able to define a readable training we collect Training Units into a Training Session to
contain some training data and results.

######_Requirements_
* Training Units as array
* Learning Gradient : floating number between 0 and 1 (but not 0 or 1)

<pre><code></code></pre>

**Task** Binding : Type

######_Business Value_
A Neuron (type) requires input data to sum and balance. This balancing value is a floating number.
Initial it must be a weight number greater or smaller than zero. But for use there is reason
to set it initial randomly between this numbers. To balance one input value we need a binding, having
at least an input neuron and a balancing property.

######_Requirements_
* input neuron as property
* balancing value as floating number property

<pre><code></code></pre>

**Task** Neuron : Type

######_Business Value_
A Neuron (type) is a unit to represent a solution for an output value and bindings to input neurons.
This bindings are nessessary to propergate input values to child neurons.
A neuron is a heart of the neuron network, it does some operations for us, but every neuron
does it the same way.

######_Requirements_
A neuron has two main operations
* calculate a new trigger value for following connected neurons in a child layer of the neural network and propagate it forward
* calculate new balancing values in input bindings based on correction delta coming propagted from a child layer neuron and propapate it back

This is the hardest part and requires further informations in a separated section of this documentation.

<pre><code></code></pre>

**Task** Layer : Type

######_Business Value_
A Layer (type) is a container for listed neurons.
It has a delegating purpose. It delegate operations through all it holding neurons.

######_Requirements_
A layer requires an listing for neurons and their creation operation.
I want to encapsule operations based on it used types, so it is easy to understand,
what this unit (layer : type) must be able to do.

## Tests

How will you train your network? With this upper prototypes your won't be able 
to describe a problem to solve with your network.

Your neural network need to be able to accept certain input values
and desired output values to orient on / to learn from it.

<pre><code></code></pre>

**Task** Training Unit : Type

######_Business Value_
A Training Unit (type) is a container for definitions of a training step.
It has 2 properties. What gets in and what is about to get out.
What gets out does not reqpresent the final values. It are desired values
for each output neuron, where the network should tweak it's weights to get to it
closely. 

######_Requirements_
Create a type to hold input values for tests and desired values for testing desired values

<pre><code></code></pre>

**Task** Training Session : Type

######_Business Value_
A Training Session (type) is a container for listed Training Units.
We should be able to hold some informations about our training collection.
Such as tests (Test Units) and the total error value.

A total error value is a representation for the quality of your network.
How fit it is (1 - total error). It is a value between 0 and 1. Zero (0) for best results
and 1 for bad results (what the heck is this?)

######_Requirements_
A Training Session required inital 2 properties:
- a list of Training Units and
- a floating property value for total error