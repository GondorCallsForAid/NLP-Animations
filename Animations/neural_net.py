import sys
import os.path
import cv2

import numpy as np

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
    def __init__(self, neural_network, stroke_width = 2, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.edge_stroke_width = stroke_width
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

    # set to manual //setup(self)
    def setup_manual(self):
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

    def construct(self):

        # manual setup of nn
        self.setup_manual()

        # display NeuralNet
        network_mob = self.network_mob.move_to([1.5,0,0]).scale(1.1)

        self.play(
           ShowCreation(network_mob)
        )
        self.wait(6)

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
        self.wait(4)


        # words as in strings are not suited as input for neural networks
        boing_pos = [-1, 1, -0.5, 0.5, -2.25]
        initial_shift = [ApplyMethod(neutral_word.shift, [2.25,0,0])]
        boing_shift = [ApplyMethod(neutral_word.shift, [pos,0,0]) for pos in boing_pos]

        # show invalid input
        self.show_activation_of_layer(layer_index = 0, activation_vector = [1,1,1], added_anims = initial_shift, valid_input = False)

        for anim in boing_shift:
            self.play(anim)


        # remove red activation
        # self.play(ApplyMethod(neutral_word.shift, [-2.25,0,0]))
        network_mob.deactivate_layers()
        self.wait(2)


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

        
        # display one-hot vectors one more time
        positive_hot_list = TextMobject("[1, 0, 0]").next_to(positive_word, DOWN)
        neutral_hot_list = TextMobject("[0, 1, 0]").next_to(neutral_word, DOWN).align_to(positive_hot_list, LEFT)
        negative_hot_list = TextMobject("[0, 0, 1]").next_to(negative_word, DOWN).align_to(positive_hot_list, LEFT)

        self.play(AnimationGroup(*[Write(positive_hot_list),
                                   Write(neutral_hot_list),
                                   Write(negative_hot_list)],
                  lag_ratio = 0.2))

        self.wait(7)

        # and add flames
        pos = [p + [0,0.15,0] for p in [positive_hot_list.get_center(), neutral_hot_list.get_center(), negative_hot_list.get_center()]]
        flame_1 = ImageMobject("images/flame.png").scale(0.4).move_to(pos[0]).shift([-0.5, 0, 0])
        flame_2 = flame_1.copy().move_to(pos[1])
        flame_3 = flame_1.copy().move_to(pos[2]).shift([0.5, 0, 0])

        flame_intro = [GrowFromCenter(flame) for flame in [flame_1, flame_2, flame_3]]
        self.play(AnimationGroup(*flame_intro))
        self.wait(1)

        flame_fadeout = [FadeOut(flame) for flame in [flame_1, flame_2, flame_3]]
        self.play(AnimationGroup(*flame_fadeout))
        #self.wait(0.5)


        # Clean up screen
        self.play(FadeOut(network_mob), FadeOut(positive_word), FadeOut(neutral_word), FadeOut(negative_word),
                  FadeOut(positive_hot_list), FadeOut(neutral_hot_list), FadeOut(negative_hot_list))

        





class EmbeddingProblem(Scene):

    def construct(self):

        # display words and their one-hot encodings
        words  = TextMobject("kept", "attention", "from", "start", "finish",
                             "great", "performances", "added", "to", "tremendous", "film").scale(0.55)

        word_num = len(words)

        encodings = [[] for i in range(word_num)]

        vectors = [Matrix([0]) for i in range(word_num)]
              

        for i in range(word_num):
            if i == 0:
                words[i].move_to([-6.1, 3.2, 0])
            else:
                words[i].next_to(words[i-1], RIGHT, buff = 0.45).align_to(words[i-1], UP)
            
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


        # better encodings - embeddings
        embeddings = [np.random.randint(low = -10, high=10, size=2) for i in range(word_num)]

        # set custom embeddings
        embeddings[3] = [-4, -2]
        embeddings[4] = [-3, -3]

        embeddings[5] = [4, -1]
        embeddings[9] = [5, -2]

        dense_anims = []
        for idx, emd in enumerate(embeddings):
            dense_anims.append(Transform(vectors[idx], Matrix(emd).scale(0.5).next_to(words[idx], DOWN)))
        self.play(AnimationGroup(*dense_anims), lag_ratio = 0.05)

        self.wait(2)


        # display embedding space
        plane_kwargs = {
        "axis_config": {
            "stroke_color": WHITE,
            "stroke_width": 2,
            "include_ticks": False,
            "include_tip": False,
            "line_to_number_buff": SMALL_BUFF,
            "label_direction": DR,
            "number_scale_val": 0.5,
        },
        "y_axis_config": {
            "label_direction": DR,
        },
        "background_line_style": {
            "stroke_color": BLUE_D,
            "stroke_width": 2,
            "stroke_opacity": 1,
        },
        # Defaults to a faded version of line_config
        "faded_line_style": None,
        "x_line_frequency": 1,
        "y_line_frequency": 1,
        "faded_line_ratio": 1,
        "make_smooth_after_applying_functions": True,
        }

        latent_space = NumberPlane(**plane_kwargs)

        self.bring_to_back(latent_space)

        self.play(GrowFromCenter(latent_space))

        self.wait(2)


        ## embed tree and flower
        fat_start = TextMobject("\\textbf{start}").move_to(words[3].get_center()).set_color(GREEN).scale(0.8)
        fat_finish = TextMobject("\\textbf{finish}").move_to(words[4].get_center()).set_color(GREEN).scale(0.8).align_to(fat_start, DOWN)
       
        highlight_anims = [Transform(words[3], fat_start), Transform(words[4], fat_finish)]
        self.play(AnimationGroup(*highlight_anims))
        self.wait(2)

        # transform embeddings into vectors
        start_vector = Vector(embeddings[3]).set_color(GREEN)
        finish_vector = Vector(embeddings[4]).set_color(GREEN)

        vector_anim = [Transform(vectors[3], start_vector),
                       Transform(vectors[4], finish_vector)]

        self.play(AnimationGroup(*vector_anim))
        self.wait(1)

        # transform vectors into points
        start_embed = Dot([-4, -2, 0]).set_color(GREEN)
        finish_embed = Dot([-3, -3, 0]).set_color(GREEN)

        def list_sum(a_list, b_list):
            return [a + b for a, b in zip(a_list, b_list)]

        embed_anim = [Transform(vectors[3], start_embed),
                      Transform(vectors[4], finish_embed),
                      ApplyMethod(words[3].move_to, list_sum(start_embed.get_center(), [0.5, 0.5, 0])),
                      ApplyMethod(words[4].move_to, list_sum(finish_embed.get_center(), [0.5, 0.5, 0]))]

        self.play(AnimationGroup(*embed_anim))

        self.wait(2)


        ## embed joyful and happy
        fat_word_3 = TextMobject("\\textbf{great}").move_to(words[5].get_center()).set_color(YELLOW).scale(0.8)
        fat_word_4 = TextMobject("\\textbf{tremendous}").move_to(words[9].get_center()).set_color(YELLOW).scale(0.8).align_to(fat_word_3, UP)
       
        highlight_anims = [Transform(words[5], fat_word_3), Transform(words[9], fat_word_4)]
        self.play(AnimationGroup(*highlight_anims))
        self.wait(2)

        # transform embeddings into vectors
        great_vector = Vector(embeddings[5]).set_color(YELLOW)
        tremendous_vector = Vector(embeddings[9]).set_color(YELLOW)

        vector_anim = [Transform(vectors[5], great_vector),
                       Transform(vectors[9], tremendous_vector)]

        self.play(AnimationGroup(*vector_anim))
        self.wait(1)


        great_embed = Dot([4, -1, 0]).set_color(YELLOW)
        tremendous_embed = Dot([5, -2, 0]).set_color(YELLOW)

        embed_anim = [Transform(vectors[5], great_embed),
                      Transform(vectors[9], tremendous_embed),
                      ApplyMethod(words[5].move_to, list_sum(great_embed.get_center(), [0.5, 0.5, 0])),
                      ApplyMethod(words[9].move_to, list_sum(tremendous_embed.get_center(), [0.5, 0.5, 0]))]

        self.play(AnimationGroup(*embed_anim))

        self.wait(4)

        # Higher dimensional embeddings
        higher_dim_e = [np.random.randint(low = -10, high=10, size=5) for i in range(word_num)]

        # exclude embeddings
        exclude = [3,4,5,9]

        higher_embeddings_anims = []
        for idx, emd in enumerate(higher_dim_e):
            if idx not in exclude:
                higher_embeddings_anims.append(Transform(vectors[idx], Matrix(emd).scale(0.5).next_to(words[idx], DOWN)))
        self.play(AnimationGroup(*higher_embeddings_anims), lag_ratio = 0.05)

        self.wait(10)





class CBOW(NetworkScene):

    def highlight_mob(self, mob, size = 1.5, color = BLUE):
        mob.generate_target()

        mob.target.scale(size).set_color(color)
        self.play(MoveToTarget(mob))

        self.wait(0.1)
       
        mob.target.scale(1/size).set_color(WHITE)
        self.play(MoveToTarget(mob))


    def construct(self):

        CBOW = TextMobject("\\textbf{CBOW}").move_to([0,2.5,0]).scale(2)
        cbow_long = TextMobject("Continuous ", "Bag-Of-Words").move_to([0,2.5,0]).scale(2)

        self.play(Write(CBOW))
        self.wait(1)

        self.play(Transform(CBOW, cbow_long))
        self.wait(1)


        ### explain meaning of "continuous" and "bag of words"

        ## continuous

        # add space
        latent_space = NumberPlane()
    
        self.play(ApplyMethod(CBOW[0].set_color, BLUE))

        self.bring_to_back(latent_space)
        self.play(GrowFromCenter(latent_space))
        self.wait(1)


        # add embeddings
        r = lambda: np.random.randint(0,255)

        dot_num = 200
        coords_and_colors = [(np.random.uniform(-6,6),             # x
                              np.random.uniform(-4,4),             # y
                              ('#%02X%02X%02X' % (r(),r(),r()))    # hex color
                             ) for i in range(dot_num)]

        dots = [Dot([x, y, 0]).set_color(color).scale(0.7) for x, y, color in coords_and_colors]

        embed_grow = [GrowFromCenter(dot) for dot in dots]
        embed_fadeout = [FadeOut(dot) for dot in dots]

        self.play(AnimationGroup(*embed_grow, lag_ratio=0.01))
        #self.wait(2)

        self.play(FadeOut(latent_space), AnimationGroup(*embed_fadeout),
                  ApplyMethod(CBOW[0].set_color, WHITE))
        #self.wait(1)


        ## bag of words
        self.play(ApplyMethod(CBOW[1].set_color, ORANGE))

        sentence = [word+" " for word in "I have never seen a movie with such great performances".split(" ")]

        bow  = TextMobject(*sentence).move_to([0,0,0]).scale(0.8)

        self.play(Write(bow))
        self.wait(2.5)


        # put words in bag
        target = 5

        bag_img = ImageMobject("images/bag.png").scale(3).move_to([-1,0,0])
        self.bring_to_back(bag_img)
        self.play(GrowFromCenter(bag_img))
        self.wait(1)

        # create bag_coords with no overlap
        bag_coords = []
        for i in range(len(bow)):
            if i == 0:
                x_cand = np.random.uniform(-2,1)
                y_cand = np.random.uniform(-2,1)
                bag_coords.append([x_cand, y_cand, 0])
                continue

            if i == target:
                bag_coords.append([5, 0, 0])

            overlap = True
            while overlap:
                x_cand = np.random.uniform(-2,1)
                y_cand = np.random.uniform(-2,1)
                
                overlap = False
                for elem in bag_coords:
                    if math.sqrt((elem[1] - y_cand)**2 + (elem[0] - x_cand)**2) < 0.8:
                        overlap = True

            bag_coords.append([x_cand, y_cand, 0])


        for i in range(len(bow)):
            if i != target:
                bow[i].generate_target()
                bow[i].target.move_to(bag_coords[i])
            else:
                bow[i].generate_target()
                bow[i].target.move_to([5, 0, 0])
                
        bag_anim = [MoveToTarget(word) for word in bow]

        # highlight target word
        self.play(ApplyMethod(bow[target].set_color, YELLOW))
        self.wait(1.5)

        # split target and context
        bow[target].generate_target()
        bow[target].target.move_to([5, 0, 0])
    
        self.play(AnimationGroup(*bag_anim, lag_ratio = 0.05),
                  MoveToTarget(bow[target]))
        self.wait(2)

        # no order shuffle
        bag_coords_2 = []
        for i in range(len(bow)):
            if i == 0:
                x_cand = np.random.uniform(-2,1)
                y_cand = np.random.uniform(-2,1)
                bag_coords_2.append([x_cand, y_cand, 0])
                continue

            if i == target:
                bag_coords_2.append([5, 0, 0])

            overlap = True
            while overlap:
                x_cand = np.random.uniform(-2,1)
                y_cand = np.random.uniform(-2,1)
                
                overlap = False
                for elem in bag_coords_2:
                    if math.sqrt((elem[1] - y_cand)**2 + (elem[0] - x_cand)**2) < 0.8:
                        overlap = True

            bag_coords_2.append([x_cand, y_cand, 0])

        for i in range(len(bow)):
            if i != target:
                bow[i].generate_target()
                bow[i].target.move_to(bag_coords_2[i])
            else:
                bow[i].generate_target()
                bow[i].target.move_to([5, 0, 0])
                
        bag_anim = [MoveToTarget(word) for word in bow]
        self.play(AnimationGroup(*bag_anim, lag_ratio = 0.05))
        self.wait(1)

        # no dependencies (highlight "great" and "performances")
        dependency_group = VGroup(bow[8], bow[9])
        self.highlight_mob(dependency_group, 1.5, BLUE)
        self.wait(1)



        # arrange words on the left side
        a = 0
        
        for i in range(len(bow)):
            if i == 0:
                bow[i].generate_target()
                bow[i].target.move_to([-6, 1.8, 0])

            elif i == target:
                bow[i].generate_target()
                bow[i].target.move_to([5, 0, 0])
                bow[i].target.set_color(YELLOW)
                a = 1
            
            else:
                bow[i].generate_target()
                bow[i].target.next_to(bow[i-1].target, DOWN).align_to(bow[i-1-a].target, LEFT)
                a = 0

        sentence_side_anim = [MoveToTarget(word) for word in bow]

        
        # move words to left side
        self.play(FadeOut(bag_img))
        self.play(AnimationGroup(*sentence_side_anim, lag_ratio = 0.05))
        self.wait(1)

        # highlight context words
        context = VGroup(*bow[:target], *bow[target+1:])
        context_summary = context.copy().scale(0.7)
        self.play(ApplyMethod(context.set_color, BLUE))
        self.wait(1)


        # build cbow model
        embedding_matrix = Matrix(["Embedding"]).move_to([-2.3,0,0]).scale(0.7)

        output_matrix = Matrix(["Output"]).move_to([1.0,0,0]).scale(0.7)

        self.play(GrowFromCenter(embedding_matrix))
        self.wait(0.5)
        self.play(GrowFromCenter(output_matrix))
        self.wait(1)
        self.play(FadeOut(output_matrix))
        self.wait(2)

        embeddings = TextMobject(*["E"+str(i) for i in range(1,10)]).move_to([1.5,1.8,0]).scale(0.6)
        embeddings.arrange(DOWN, aligned_edge=LEFT, center = False, buff=0.30)

        # transform all at once
        # self.play(Transform(context.copy(), embeddings))

        # individual transform through matrix
        anims = []

        for word, embed in zip(context, embeddings):
            matrix_pos_diff = embedding_matrix.get_center() - word.get_center()
            anims.append(ApplyMethod(word.shift, matrix_pos_diff))
            anims.append(Transform(word, embed))
            
        for anim in anims:
            self.play(anim)

        self.wait(1)

        # shift embedding matrix and embeddings to the left
        self.play(ApplyMethod(context.shift, [-2.7, 0, 0]),
                  ApplyMethod(embedding_matrix.shift, [-2.7, 0, 0]))
        self.wait(1)


        # sum embeddings 
        sum_symbol = TextMobject("$\\sum$").scale(1).next_to(embedding_matrix, RIGHT).shift([1.7,0.0,0])
        self.play(Write(sum_symbol))
        #self.wait(1)

        e_sum = TextMobject("$E_{sum}$").move_to(sum_symbol.get_center())

        self.play(Transform(context, e_sum), FadeOut(sum_symbol))
        #self.wait(1)

        # push sum through output matrix
        self.play(ApplyMethod(embedding_matrix.shift, [-4, 0, 0]),
                  ApplyMethod(context.shift, [-2, 0, 0]))

        output_matrix = Matrix(["Output"]).next_to(context, RIGHT).scale(0.7)
        self.play(GrowFromCenter(output_matrix))
        #self.wait(0.5)

        e_out = TextMobject("$E_{out}$").next_to(output_matrix, RIGHT).shift([0.5, 0, 0])
        self.play(Transform(context, e_out))
        #self.wait(0.5)

        # apply softmax
        self.play(ApplyMethod(output_matrix.shift, [-10, 0, 0]))

        e_out_soft = TextMobject("$Softmax(E_{out})$").move_to(e_out.get_center())
        self.play(Transform(context, e_out_soft))
        self.wait(2)

        # compare probs
        soft_probs = Matrix([.01, .02, .01, .01, "0.90", .01, .01, .01, .02]).scale(0.5)
        soft_probs.move_to(context.get_center()).shift([1.5, 0, 0])

        movie_one_hot = Matrix([0, 0, 0, 0, 1, 0, 0, 0, 0]).set_color(YELLOW).scale(0.5)
        movie_one_hot.next_to(bow[target], LEFT).shift([-0.5, 0, 0])

        self.play(Transform(context, soft_probs))
        self.wait(0.5)

        # highlight highest prob, add square

        rect = Rectangle(height = 0.4, width = 0.55).move_to(soft_probs.get_entries()[4].get_center())
        rect.set_color(BLUE)
        high_prob = soft_probs.get_entries()[4]
        self.play(ApplyMethod(high_prob.set_color, BLUE),
                  GrowFromCenter(rect))
        self.wait(5)

        self.play(GrowFromCenter(movie_one_hot))
        self.wait(3)

        # get loss
        loss = TextMobject("Loss").move_to((context.get_center() + movie_one_hot.get_center())/2).shift([2,0,0])
        tmp_group = VGroup(context, movie_one_hot)

        tmp_group_copy = tmp_group.copy()
        self.play(FadeOut(bow[target]), Transform(tmp_group_copy, loss), FadeOut(rect), FadeOut(high_prob))
        self.wait(8)


        # blend in matrices
        embedding_matrix = Matrix(["Embedding"]).move_to([-3,0,0]).scale(0.7)
        output_matrix = Matrix(["Output"]).next_to(embedding_matrix, RIGHT).scale(0.7)
        self.play(GrowFromCenter(embedding_matrix),
                  GrowFromCenter(output_matrix))
        self.wait(2)


        # clear screen
        self.play(FadeOut(embedding_matrix), FadeOut(output_matrix), FadeOut(tmp_group), FadeOut(tmp_group_copy))

        # summarize
        self.play(GrowFromCenter(context_summary))

        embedding_matrix = Matrix(["Embedding"]).move_to([-3,0,0]).scale(0.6)
        self.play(GrowFromCenter(embedding_matrix))

        sum_symbol = TextMobject("$\\sum$").scale(0.6).next_to(embedding_matrix, RIGHT).shift([0.5, 0, 0])
        self.play(GrowFromCenter(sum_symbol))

        output_matrix = Matrix(["Output"]).next_to(sum_symbol, RIGHT).scale(0.6)
        self.play(GrowFromCenter(output_matrix))

        softmax = TextMobject("$Softmax()$").next_to(output_matrix, RIGHT).scale(0.6)
        self.play(GrowFromCenter(softmax))

        target_word = bow[target].next_to(softmax, RIGHT).shift([0.5, 0.05, 0])
        self.play(GrowFromCenter(target_word))


        # highlight embeddings matrix
        self.highlight_mob(embedding_matrix, 1.5, BLUE)
        self.wait(10)

        # highlight embeddings matrix again
        self.highlight_mob(embedding_matrix, 1.5, BLUE)
        self.wait(8)

        # clear screen
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )



class DualNetwork(NetworkScene):

    def construct(self):

        # if we want to do sentiment analysis
        emotions = ["happy", "neutral", "sad"]
        smilies = [ImageMobject("images/" + emotion + ".png").move_to([5,0,0]).scale(0.8) for emotion in emotions]
        positions = [[-3, 0, 0], [0, 0, 0], [3, 0, 0]]

        anims = []

        for smiley, pos in zip(smilies, positions):
            smiley.move_to(pos)

        self.wait(1)
        
        self.play(GrowFromCenter(smilies[0]),
                  GrowFromCenter(smilies[1]),
                  GrowFromCenter(smilies[2]))
        self.wait(0.5)

        self.play(FadeOut(smilies[0]),
                  FadeOut(smilies[1]),
                  FadeOut(smilies[2]))
        self.wait(0.5)


        embed_net_label = TextMobject("Embedding Net").move_to([-2.65, 3, 0])
        embedding_net = Network(sizes = [9, 4])
        embedding_net_mob = NetworkMobject(embedding_net).move_to([-2.5, 0, 0]).scale(0.9)

        class_net_label = TextMobject("Classifyer Net").move_to([2.1, 3, 0])
        class_net_label_tmp = class_net_label.copy()
        classifier_net = Network(sizes = [4, 8, 8, 8, 2])
        classifier_net_mob = NetworkMobject(classifier_net).move_to([2.5, 0, 0]).scale(0.9)

        self.play(GrowFromCenter(embedding_net_mob), Write(embed_net_label))
        self.wait(1.5)
        self.play(GrowFromCenter(classifier_net_mob), Write(class_net_label))
        self.wait(2)

        dual_network = TextMobject("Dual Network").move_to([0, 3, 0])

        embed_class_group = VGroup(embed_net_label, class_net_label)

        self.play(Transform(embed_class_group, dual_network))
        self.wait(6)

        self.play(Transform(embed_class_group, class_net_label_tmp))
        self.wait(0.5)

        self.play(ApplyMethod(embedding_net_mob.shift, [-7, 0, 0]),
                  ApplyMethod(classifier_net_mob.shift, [-2.5, 0, 0]),
                  ApplyMethod(embed_class_group.shift, [-2.5, 0, 0]))
        
        # clear screen
        self.play(FadeOut(embedding_net_mob), FadeOut(classifier_net_mob), FadeOut(embed_class_group))



class SingularNetwork(NetworkScene, GraphScene):

    CONFIG = {
        "graph_origin" : [-5, -3, 0],
        "y_max" : 5,
        "y_min" : 0,
        "x_max" : 7,
        "x_min" : 0,
        "y_tick_frequency" : 1, 
        "x_tick_frequency" : 1, 
        "axes_color" : WHITE, 
        "y_labeled_nums": range(0,60,10),
        #"x_labeled_nums": list(np.arange(2, 7.0+0.5, 0.5)),
        "x_label_decimal":1,
        "y_label_direction": RIGHT,
        "x_label_direction": UP,
        "y_label_decimal":3,
        "y_axis_label": "Loss",
        "x_axis_label": "Epoch"

    }

    def construct(self):

        embed_net_label = TextMobject("Embedding Net").move_to([-2.65, 3, 0])
        embedding_net = Network(sizes = [9, 4])
        embedding_net_mob = NetworkMobject(embedding_net).move_to([-2.5, 0, 0]).scale(0.9)

        class_net_label = TextMobject("Classifyer Net").move_to([2.1, 3, 0])
        classifier_net = Network(sizes = [4, 8, 8, 8, 2])
        classifier_net_mob = NetworkMobject(classifier_net).move_to([2.5, 0, 0]).scale(0.9)

        self.play(GrowFromCenter(embedding_net_mob), Write(embed_net_label),
                  GrowFromCenter(classifier_net_mob), Write(class_net_label))
        self.wait(14)

        dual_network = VGroup(embedding_net_mob, classifier_net_mob)
        dual_label = VGroup(embed_net_label, class_net_label)

        edge_stroke_width_overwrite = True
        overwrite_width = 1

        singular_net_label = TextMobject("Singular Net").move_to([0, 3, 0])
        singular_net = Network(sizes = [9, 12, 12, 12, 8])
        singular_net_mob = NetworkMobject(singular_net, stroke_width = 1).move_to([0, 0, 0]).scale(0.7)

        out_net_embed = Network(sizes = [8, 4])
        out_net_embed_mob = NetworkMobject(out_net_embed, stroke_width = 1).move_to([2.26, 0, 0]).scale(0.7)
        out_net_embed_mob_copy = out_net_embed_mob.copy()

        out_net_class = Network(sizes = [8, 2])
        out_net_class_mob = NetworkMobject(out_net_class, stroke_width = 1).move_to([2.26, 0, 0]).scale(0.7)

        singular_net_group = VGroup(singular_net_mob, out_net_embed_mob)


        # Dual to singular
        self.play(Transform(dual_network, singular_net_group), 
                  Transform(dual_label, singular_net_label))
        self.wait(2)


        # pre training
        embedding = Matrix(np.random.randint(low = -10, high=10, size=4)).next_to(dual_network, RIGHT)
        self.play(GrowFromCenter(embedding))
        self.wait(3)
        self.play(FadeOut(embedding))
        self.wait(1)


        ## fine tuning
        # replace embedding out with class out
        self.play(Transform(dual_network[1], out_net_class_mob))

        # output sentiments
        emotions = ["sad", "neutral", "happy"]
        smilies = [ImageMobject("images/" + emotion + ".png").move_to([4,0,0]).scale(0.8) for emotion in emotions]

        for smiley in smilies:
            self.play(GrowFromCenter(smiley))
            self.wait(0.5)
            self.play(FadeOut(smiley))

        self.wait(27)

        # clean screen
        #self.play(FadeOut(dual_network), FadeOut(dual_label))


        self.play(Transform(dual_network[1], out_net_embed_mob_copy))
        self.wait(9)

        self.play(ApplyMethod(dual_network.scale, 1.2))
        self.play(ApplyMethod(dual_network.scale, 1/1.2))
        self.wait(3)

        self.play(FadeOut(dual_network), FadeOut(dual_label))


        # show overfitting
        up_right_corner = [4, 3, 0]     

        train_loss_label = TexMobject("Train Loss", color = BLUE).move_to(up_right_corner)
        test_loss_label = TexMobject("Test Loss", color = ORANGE).next_to(train_loss_label, DOWN)

        self.play(
            ShowCreation(train_loss_label),
            ShowCreation(test_loss_label)
        )
        self.setup_axes(animate=True)


        train_loss = self.get_graph(lambda x : 1/(x+0.1),  
                                    color = BLUE,
                                    x_min = 0.1, 
                                    x_max = 7
                                    )

        test_loss_2 = self.get_graph(lambda x : 1/(x+0.1) + (x*0.2)**3,  
                                    color = ORANGE,
                                    x_min = 0.1, 
                                    x_max = 7
                                    )

        self.play(
            ShowCreation(train_loss),
            ShowCreation(test_loss_2),
            run_time = 4
        )


        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        # restore singular network
        self.play(GrowFromCenter(dual_network), GrowFromCenter(dual_label))
        self.wait(20)

        books = ImageMobject("images/books.png").move_to([-4, 1.7, 0])
        wikipedia = ImageMobject("images/wikipedia.png").move_to([-4, -1.7, 0])

        self.play(GrowFromCenter(books))
        self.play(GrowFromCenter(wikipedia))
        self.wait(3)

        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        bert = TextMobject("BERT").scale(3).move_to([0,0,0])
        self.play(GrowFromCenter(bert))
        self.wait(1)

        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )












    
        
