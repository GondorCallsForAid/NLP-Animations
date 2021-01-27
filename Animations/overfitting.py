from manimlib.imports import *

class Plot1(GraphScene):
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
            run_time = 5
        )
        self.wait(1)