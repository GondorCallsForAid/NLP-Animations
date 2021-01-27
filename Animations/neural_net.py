import sys
import os.path
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from manimlib.imports import *

from Animations.nn.network import *

class NetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius" : 0.15,
        "neuron_to_neuron_buff" : MED_SMALL_BUFF,
        "layer_to_layer_buff" : LARGE_BUFF,
        "neuron_stroke_color" : BLUE,
        "neuron_stroke_width" : 3,
        "neuron_fill_color" : GREEN,
        "edge_color" : LIGHT_GREY,
        "edge_stroke_width" : 2,
        "edge_propogation_color" : YELLOW,
        "edge_propogation_time" : 1,
        "max_shown_neurons" : 16,
        "brace_for_large_layers" : True,
        "average_shown_activation_of_large_layer" : True,
        "include_output_labels" : False,
    }
    def __init__(self, neural_network, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.neural_network = neural_network
        self.layer_sizes = neural_network.sizes
        self.add_neurons()
        self.add_edges()

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size)
            for size in self.layer_sizes
        ])
        layers.arrange(RIGHT, buff = self.layer_to_layer_buff)
        self.layers = layers
        self.add(self.layers)
        if self.include_output_labels:
            self.add_output_labels()

    def get_layer(self, size):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
                radius = self.neuron_radius,
                stroke_color = self.neuron_stroke_color,
                stroke_width = self.neuron_stroke_width,
                fill_color = self.neuron_fill_color,
                fill_opacity = 0,
            )
            for x in range(n_neurons)
        ])   
        neurons.arrange(
            DOWN, buff = self.neuron_to_neuron_buff
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = TexMobject("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    def get_edge(self, neuron1, neuron2):
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff = self.neuron_radius,
            stroke_color = self.edge_color,
            stroke_width = self.edge_stroke_width,
        )

    def get_active_layer(self, layer_index, activation_vector, valid_input = True):
        layer = self.layers[layer_index].deepcopy()
        self.activate_layer(layer, activation_vector, valid_input)
        return layer

    def activate_layer(self, layer, activation_vector, valid_input = True):
        n_neurons = len(layer.neurons)
        av = activation_vector
        def arr_to_num(arr):
            return (np.sum(arr > 0.1) / float(len(arr)))**(1./3)

        if len(av) > n_neurons:
            if self.average_shown_activation_of_large_layer:
                indices = np.arange(n_neurons)
                indices *= int(len(av)/n_neurons)
                indices = list(indices)
                indices.append(len(av))
                av = np.array([
                    arr_to_num(av[i1:i2])
                    for i1, i2 in zip(indices[:-1], indices[1:])
                ])
            else:
                av = np.append(
                    av[:n_neurons/2],
                    av[-n_neurons/2:],
                )
        for activation, neuron in zip(av, layer.neurons):
            if not valid_input:
                self.neuron_fill_color = "#e71212"
            else:
                self.neuron_fill_color = GREEN

            neuron.set_fill(
                color = self.neuron_fill_color,
                opacity = activation
            )
        return layer

    def activate_layers(self, input_vector, valid_input = True):
        activations = self.neural_network.get_activation_of_all_layers(input_vector)
        for activation, layer in zip(activations, self.layers):
            self.activate_layer(layer, activation, valid_input)

    def deactivate_layers(self):
        all_neurons = VGroup(*it.chain(*[
            layer.neurons
            for layer in self.layers
        ]))
        all_neurons.set_fill(opacity = 0)
        return self

    def get_edge_propogation_animations(self, index):
        edge_group_copy = self.edge_groups[index].copy()
        edge_group_copy.set_stroke(
            self.edge_propogation_color,
            width = 1.5*self.edge_stroke_width
        )
        return [ShowCreationThenDestruction(
            edge_group_copy, 
            run_time = self.edge_propogation_time,
            lag_ratio = 0.5
        )]

    def add_output_labels(self):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = TexMobject(str(n))
            label.set_height(0.75*neuron.get_height())
            label.move_to(neuron)
            label.shift(neuron.get_width()*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)


class NetworkScene(Scene):
    CONFIG = {
        "layer_sizes" : [3, 6, 6, 2],
        "network_mob_config" : {},
    }
    def setup(self):
        self.add_network()

    def add_network(self):
        self.network = Network(sizes = self.layer_sizes)
        self.network_mob = NetworkMobject(
            self.network,
            **self.network_mob_config
        )
        self.add(self.network_mob)

    def feed_forward(self, input_vector, false_confidence = False, added_anims = None, stop_at_layer = None):
        if added_anims is None:
            added_anims = []
        activations = self.network.get_activation_of_all_layers(
            input_vector
        )
        if false_confidence:
            i = np.argmax(activations[-1])
            activations[-1] *= 0
            activations[-1][i] = 1.0
        for i, activation in enumerate(activations):
            if i == stop_at_layer:
                break
            self.show_activation_of_layer(i, activation, added_anims)
            added_anims = []

    def show_activation_of_layer(self, layer_index, activation_vector, added_anims = None, valid_input = True):
        if added_anims is None:
            added_anims = []
        layer = self.network_mob.layers[layer_index]
        active_layer = self.network_mob.get_active_layer(
            layer_index, activation_vector, valid_input
        )
        anims = [Transform(layer, active_layer)]
        if layer_index > 0:
            anims += self.network_mob.get_edge_propogation_animations(
                layer_index-1
            )
        anims += added_anims
        self.play(*anims)

    def remove_random_edges(self, prop = 0.9):
        for edge_group in self.network_mob.edge_groups:
            for edge in list(edge_group):
                if np.random.random() < prop:
                    edge_group.remove(edge)




class IntroduceNetwork(NetworkScene):

    def boing_shift(self, mob, direction = -1):
        pos = mob.get_center()



    def construct(self):

        # display NeuralNet
        network_mob = self.network_mob.move_to([1.5,0,0]).scale(1.1)

        self.play(
           ShowCreation(network_mob)
        )

        # display words
        positive_word = TextMobject("\\textbf{amazing}").move_to([-4.5,2,0]).set_color(GREEN)

        neutral_word = TextMobject("\\textbf{waterfall}").set_color(YELLOW)
        neutral_word.move_to([-4.5,0,0]).align_to(positive_word, LEFT)

        negative_word = TextMobject("\\textbf{terrible}").set_color(RED)
        negative_word.move_to([-4.5,-2,0]).align_to(positive_word, LEFT)


        self.play(AnimationGroup(*[Write(positive_word),
                                   Write(neutral_word),
                                   Write(negative_word)],
                  lag_ratio = 1))
        self.wait(2)


        # words as in strings are not suited as input for neural networks
        boing_pos = [-1, 1, -0.5, 0.5]
        initial_shift = [ApplyMethod(neutral_word.shift, [2.25,0,0])]
        boing_shift = [ApplyMethod(neutral_word.shift, [pos,0,0]) for pos in boing_pos]

        # show invalid input
        self.show_activation_of_layer(layer_index = 0, activation_vector = [1,1,1], added_anims = initial_shift, valid_input = False)

        for anim in boing_shift:
            self.play(anim)


        # shift word back and remove red activation
        self.play(ApplyMethod(neutral_word.shift, [-2.25,0,0]))
        network_mob.deactivate_layers()
        self.wait(1)


        # display one-hot vectors of words
        positive_hot_list = TextMobject("[1, 0, 0]").next_to(positive_word, DOWN)
        neutral_hot_list = TextMobject("[0, 1, 0]").next_to(neutral_word, DOWN).align_to(positive_hot_list, LEFT)
        negative_hot_list = TextMobject("[0, 0, 1]").next_to(negative_word, DOWN).align_to(positive_hot_list, LEFT)


        self.play(AnimationGroup(*[Write(positive_hot_list),
                                   Write(neutral_hot_list),
                                   Write(negative_hot_list)],
                  lag_ratio = 1))
        self.wait(2)


        ## pass one-hot vectors to network
        encodings = [[1,0,0], [0,1,0], [0,0,1]]
        hot_lists = [positive_hot_list, neutral_hot_list, negative_hot_list]
        outputs = [[1,0], [1,1], [0,1]]

        emotions = ["happy", "neutral", "sad"]
        smilies = [ImageMobject("images/" + emotion + ".png").move_to([5,0,0]).scale(0.8) for emotion in emotions]

        for encoding, hot_list, output, smiley in zip(encodings, hot_lists, outputs, smilies):
            network_mob.deactivate_layers()
            matrix = Matrix(encoding).move_to([-1.5,0,0]).scale(0.80)
            self.play(Transform(hot_list, matrix))
            self.feed_forward(input_vector = np.array(encoding), stop_at_layer = 3)
            self.show_activation_of_layer(layer_index = 3, activation_vector = output)
            self.play(GrowFromCenter(smiley))
            self.wait(1)
            self.play(FadeOut(hot_list))
            self.wait(1)
            self.play(FadeOut(smiley))

        

        # Clean up screen
        self.play(FadeOut(network_mob), FadeOut(positive_word), FadeOut(neutral_word), FadeOut(negative_word))




class EmbeddingProblem(Scene):

    def construct(self):

        # display words and their one-hot encodings
        words  = TextMobject("amazing", "waterfall", "terrible", "flower", "tower",
                             "house", "tree", "joyful", "blue", "book", "happy").scale(0.5)

        encodings = [[] for i in range(len(words))]

        vectors = [Matrix([0]) for i in range(len(encodings))]
              

        for i in range(len(words)):
            if i == 0:
                words[i].move_to([-5.6, 3, 0])
            else:
                words[i].next_to(words[i-1], RIGHT, buff = 0.5).align_to(words[i-1], UP)
            
            # write word
            self.play(Write(words[i]))

            # update encodings
            trans_anims = []
            for idx, e in enumerate(encodings):
                if i == idx:
                    e.append(1)
                else:
                    e.append(0)
                # update vector representations
                if idx <= i:
                    trans_anims.append(Transform(vectors[idx], Matrix(e).scale(0.5).next_to(words[idx], DOWN)))

            self.play(AnimationGroup(*trans_anims), lag_ratio = 0.05)

        self.wait(2)



#class 



    
        
