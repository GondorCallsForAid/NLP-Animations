from manimlib.imports import *
import numpy as np
import math


class Intro(Scene):

    def construct(self):

        sa = TextMobject('Sentiment Analysis').set_color_by_gradient(RED, YELLOW, GREEN).scale(2.2)
        self.wait(0.5)
        self.play(DrawBorderThenFill(sa))
        self.wait(2)

class Applications(Scene):

    def construct(self):

        apps = TextMobject('Applications').scale(2)
        self.wait(0.5)
        self.play(DrawBorderThenFill(apps))
        self.wait(1)

        self.play(ApplyMethod(apps.shift, [0,2.75,0]))
        self.wait(1)

        ## amazon logo
        amz = ImageMobject("images/amazon_logo_white.png").move_to([-4, 1, 0]).scale(0.5)
        self.play(GrowFromCenter(amz))
        self.wait(1)

        # amazon reviews
        amz_rev_1 = ImageMobject("images/amazon_review_1.png").move_to([-4, -0.5, 0]).scale(0.5)
        amz_rev_2 = ImageMobject("images/amazon_review_2.png").move_to([-4, -1.6, 0]).scale(0.491)
        amz_rev_3 = ImageMobject("images/amazon_review_3.png").move_to([-4, -2.58, 0]).scale(0.385)
        self.play(GrowFromCenter(amz_rev_1))
        self.play(GrowFromCenter(amz_rev_2))
        self.play(GrowFromCenter(amz_rev_3))
        self.wait(1)


        ## twitter logo
        twitter = ImageMobject("images/twitter_logo.png").move_to([0, 1.25, 0]).scale(0.8)
        self.play(GrowFromCenter(twitter))
        self.wait(1)

        # twitter posts
        twt_post_1 = ImageMobject("images/twt_post_1.png").move_to([0, -0.8, 0]).scale(0.8) 
        twt_post_2 = ImageMobject("images/twt_post_2.png").move_to([0, -2.1, 0]).scale(0.375) 
        twt_post_3 = ImageMobject("images/twt_post_3.png").move_to([0, -3.08, 0]).scale(0.495)
        self.play(GrowFromCenter(twt_post_1))
        self.play(GrowFromCenter(twt_post_2))
        self.play(GrowFromCenter(twt_post_3))
        self.wait(1) 


        ## facebook logo
        fb = ImageMobject("images/facebook_logo.png").move_to([4, 1.25, 0]).scale(0.8)
        self.play(GrowFromCenter(fb))
        self.wait(1)

        # facebook posts
        fb_post_1 = ImageMobject("images/fb_post_1.png").move_to([4, -0.66, 0]).scale(0.66)
        fb_post_2 = ImageMobject("images/fb_post_2.png").move_to([4, -1.63, 0]).scale(0.733)
        fb_post_3 = ImageMobject("images/fb_post_3.png").move_to([4, -3.052, 0]).scale(0.571)
        self.play(GrowFromCenter(fb_post_1))
        self.play(GrowFromCenter(fb_post_2))
        self.play(GrowFromCenter(fb_post_3))
        self.wait(1) 


        # make everything transparent
        on_screen = [apps, amz, amz_rev_1, amz_rev_2, amz_rev_3, twitter, twt_post_1, twt_post_2, twt_post_3, 
                     fb, fb_post_1, fb_post_2, fb_post_3]

        tmp_anims = [ApplyMethod(elem.set_opacity, 0.2) for elem in on_screen]
        self.play(AnimationGroup(*tmp_anims))


        # let "many more things" fly over screen
        wikipedia = ImageMobject("images/wikipedia.png").move_to([8.5, 0, 0])
        imdb = ImageMobject("images/imdb_logo.png").move_to([11.5, 0, 0]) 
        books = ImageMobject("images/books.png").move_to([14.5, 0, 0])
        rt = ImageMobject("images/rt_logo.png").move_to([17.5, 0, 0])
        yt = ImageMobject("images/yt_logo.png").move_to([20.5, 0, 0])

        self.play(ApplyMethod(wikipedia.shift, [-31, 0, 0]),
                  ApplyMethod(imdb.shift, [-31, 0, 0]),
                  ApplyMethod(books.shift, [-31, 0, 0]),
                  ApplyMethod(rt.shift, [-31, 0, 0]),
                  ApplyMethod(yt.shift, [-31, 0, 0]), run_time = 6)
    

        # clear screen
        tmp_anims = [FadeOut(elem) for elem in on_screen]
        self.play(AnimationGroup(*tmp_anims))
        self.wait(0.5)



class PrePostNN(Scene):

    def construct(self):

        pre = ImageMobject("images/stone_age_cut.png").move_to([-10, 0, 0]).scale(4)
        post = ImageMobject("images/future_cut.png").move_to([10, 0, 0]).scale(4)

        self.wait(0.5)
        self.play(ApplyMethod(pre.shift, [10, 0, 0]))
        self.wait(0.5)
        self.play(ApplyMethod(post.shift, [-9.9, 0, 0]))
        self.wait(1)

        ## list topcis
        # pre neural network
        pre_neural = TextMobject("Dictionary-based", "Corpus-based", "Naive Bayes").move_to([-0.5, 2.2 ,0]).set_color(BLACK)
        pre_neural.arrange(DOWN, aligned_edge=LEFT, center = False, buff=0.50).scale(1.5)

        # post neural network
        post_neural = TextMobject("Embeddings", "Transformer", "BERT", "XLNet").move_to([7, 2, 0]).set_color(BLACK)
        post_neural[3].set_color(WHITE)
        post_neural.arrange(DOWN, aligned_edge=LEFT, center = False, buff=0.50).scale(1.5)

        self.play(Write(pre_neural)) 
        self.play(Write(post_neural[0:3]))
        self.wait(1)
        self.play(Write(post_neural[3]))
        self.wait(2)

        # clear screen
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )



def rnd_circle_coords_and_color(radius = 2, x_center = 0, y_center = 0):
    # random angle
    alpha = 2 * math.pi * np.random.uniform()
    # random radius (for uniform dist)
    u = np.random.uniform() + np.random.uniform()
    r = radius * (2 - u if u > 1 else u)
    # calculating coordinates
    x = r * math.cos(alpha) + x_center
    y = r * math.sin(alpha) + y_center

    # get color
    cols = [GREEN, YELLOW, RED]
    col = cols[np.random.randint(3)]

    return x, y, col


def calc_shift(embedding, x_center, y_center):

    embed_pos = np.array(embedding.get_center())
    clusters = {"#83c167": [-3, 1, 0], "yellow": [0, -2, 0], "#fc6255": [3, 1, 0]}
    center_diff =  embed_pos - np.array([x_center, y_center, 0])
    try:
        cluster = clusters[str(embedding.get_color())]
    except KeyError:
        print(str(embedding.get_color()))
    target_pos = cluster + center_diff
    shift = target_pos - embed_pos

    return shift



class FineTuning(Scene):

    def construct(self):

        # load embedding space
        latent_space = NumberPlane()
        self.play(GrowFromCenter(latent_space))

        # embedding labels
        pos = Dot([-5, 3, 0]).set_color(GREEN).scale(2)
        pos_label = TextMobject("positive").set_color(GREEN).next_to(pos, RIGHT)

        ntrl = Dot([-1, 3, 0]).set_color(YELLOW).scale(2)
        ntrl_label = TextMobject("neutral").set_color(YELLOW).next_to(ntrl, RIGHT)

        neg = Dot([3, 3, 0]).set_color(RED).scale(2)
        neg_label = TextMobject("negative").set_color(RED).next_to(neg, RIGHT)

        self.play(GrowFromCenter(pos), GrowFromCenter(ntrl), GrowFromCenter(neg),
                  Write(pos_label), Write(ntrl_label), Write(neg_label))


        # generate positive, neutral, negative embeddings together in circle
        embed_num = 300
        radius, x_center, y_center = 2, 0, -1

        embeddings = [Dot([x, y, 0]).set_color(col) for x,y,col in [rnd_circle_coords_and_color(radius, x_center, y_center) for i in range(embed_num)]]

        tmp_anims = [GrowFromCenter(e) for e in embeddings]
        self.play(AnimationGroup(*tmp_anims, lag_ratio = 0.01))
        self.wait(1)

        # split embeddings 
        shift = [calc_shift(e, x_center, y_center) for e in embeddings]

        tmp_anims = [ApplyMethod(e.shift, v) for e, v in zip(embeddings, shift)]
        self.play(AnimationGroup(*tmp_anims, lag_ratio = 0.005))

        # clear screen
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        self.wait(0.5)
    


class BERT(Scene):

    def construct(self):

        bert = TextMobject('B', 'E', 'R', 'T').scale(2)
        self.wait(0.5)
        self.play(DrawBorderThenFill(bert))
        self.wait(1)

        # shift BERT up
        self.play(ApplyMethod(bert.shift, [0, 3, 0]))
        self.wait(1)


        ## T
        transformer = TextMobject('Transformer').scale(1.5).set_color(BLUE)
        self.wait(0.5)
        self.play(DrawBorderThenFill(transformer))
        self.wait(1)

        # hightlight T
        bert[3].generate_target()

        bert[3].target.scale(1.5).set_color(BLUE)
        self.play(MoveToTarget(bert[3]))
        self.wait(0.1)

        bert[3].target.scale(1/1.5)
        self.play(MoveToTarget(bert[3]))
        self.wait(1)

        # shift transformer up and display images
        self.play(ApplyMethod(transformer.shift, [0,2,0]))
        encoder_img = ImageMobject("images/encoder.png").scale(1.6).move_to([-0.896, -2.17, 0])
        decoder_img = ImageMobject("images/decoder.png").scale(2.8).move_to([0.64, -1.2, 0])
        self.play(GrowFromCenter(encoder_img), GrowFromCenter(decoder_img))
        self.wait(1)


        ## E
        enc_dec_struct = TextMobject('Encoder', '+', 'Decoder').scale(1.5)
        enc_dec_struct[0].set_color(YELLOW)
        enc_dec_struct[0].move_to([-3, 2, 0])
        enc_dec_struct[1].move_to([0, 2, 0])
        enc_dec_struct[2].move_to([3, 2, 0])
        self.play(Transform(transformer, enc_dec_struct), 
                  ApplyMethod(encoder_img.shift, [-2, 0, 0]),
                  ApplyMethod(decoder_img.shift, [2, 0, 0]))
        self.wait(1)


        # highlight E
        bert[1].generate_target()

        bert[1].target.scale(1.5).set_color(YELLOW)
        self.play(MoveToTarget(bert[1]))
        self.wait(0.1)

        bert[1].target.scale(1/1.5)
        self.play(MoveToTarget(bert[1]))
        self.wait(1)

        self.play(FadeOut(transformer[1]), FadeOut(transformer[2]), FadeOut(decoder_img))

        # stack encoders to get the ultimate embeddings

        self.play(ApplyMethod(encoder_img.shift, [-1.2, 2, 0]))
        self.play(encoder_img.rotate, -PI/2, about_point = encoder_img.get_center())

        encoder_block = ImageMobject("images/encoder_block.png").scale(1.6).move_to([-2, -0.2, 0])
        encoder_block.rotate(-PI/2, about_point = encoder_block.get_center())

        encoder_block_2 = encoder_block.copy().shift([2, 0, 0])
        encoder_block_3 = encoder_block_2.copy().shift([2, 0, 0])
        encoder_block_4 = encoder_block_3.copy().shift([2, 0, 0])

        self.play(GrowFromCenter(encoder_block))
        self.play(GrowFromCenter(encoder_block_2))
        self.play(GrowFromCenter(encoder_block_3))
        self.play(GrowFromCenter(encoder_block_4))

        self.wait(1)

        # normal embedding
        normal_embedding = Matrix(np.random.randint(low = 0, high=10, size=5)).next_to(encoder_img, LEFT)
        normal_embedding.scale(0.7).shift([0.1, 0, 0])
        self.play(GrowFromCenter(normal_embedding))
        self.wait(2)


        # ultimate embedding
        ultimate_embedding = Matrix(np.random.randint(low = 0, high=10, size=5)).next_to(encoder_block_4, RIGHT)
        ultimate_embedding.set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE).scale(0.7).shift([-0.1, 0, 0])
        #ultimate_embedding.set_color_by_gradient(WHITE, YELLOW_E)

        #self.play(DrawBorderThenFill(ultimate_embedding))
        tmp_dot = Dot([0, 0, 0]).set_opacity(0)
        self.play(Transform(normal_embedding, tmp_dot))
        self.play(Transform(normal_embedding, ultimate_embedding))
        self.wait(1)

        # highlight BERT
        bert.generate_target()
        bert.target.scale(1.5)
        self.play(MoveToTarget(bert))
        self.wait(0.1)
        bert.target.scale(1/1.5)
        self.play(MoveToTarget(bert))
        self.wait(2)

        # highlight attention blocks
        mha_square = Rectangle(height=1.15, width=2.5, color = "#fde3bb", fill_opacity=1).move_to([0, -2.5, 0])
        mha_multi_head = TextMobject("Multi-Head", color = BLACK).move_to(mha_square.get_center()).shift([0, 0.25, 0]).scale(0.9)
        mha_attention = TextMobject("attention", color = BLACK).next_to(mha_multi_head, DOWN).shift([0, 0.1, 0]).scale(0.9)
        mha_block = VGroup(mha_square, mha_multi_head, mha_attention)
        mha_block.rotate(-PI/2, about_point = mha_block.get_center())

        self.play(GrowFromCenter(mha_block))
        self.wait(1)

        self.play(mha_block.rotate, PI/2, about_point = mha_block.get_center())
        self.wait(1)

        ### explain multi-head-attention
        self.play(FadeOut(normal_embedding), FadeOut(encoder_img), FadeOut(encoder_block), FadeOut(encoder_block_2), 
                  FadeOut(encoder_block_3), FadeOut(encoder_block_4), FadeOut(bert), FadeOut(transformer[0]))

        mha_text = VGroup(mha_multi_head, mha_attention)
        mha_text.generate_target()
        mha_text.target.move_to([0, 2.9, 0]).set_color(WHITE).scale(1.5)

        self.play(FadeOut(mha_square), MoveToTarget(mha_text))
        self.wait(2)   

        ## building blocks
        # normal embeddings
        green_big = Rectangle(height=0.5, width=2, color = GREEN_B, fill_opacity=1)
        x_pos = [0, 0.5, 1, 1.5]
        green_small = [Rectangle(height=0.5, width=0.5, color = GREEN_E).shift([-0.75 + x, 0, 0]) for x in x_pos]
        n_embed_block = VGroup(green_big , *green_small).move_to([0, 2, 0])

        # key
        orange_big = Rectangle(height=0.5, width=2, color = GOLD_B, fill_opacity=1)
        x_pos = [0, 0.5, 1, 1.5]
        orange_small = [Rectangle(height=0.5, width=0.5, color = ORANGE).shift([-0.75 + x, 0, 0]) for x in x_pos]
        key_block = VGroup(orange_big , *orange_small).move_to([0, 1, 0])

        # query
        purple_big = Rectangle(height=0.5, width=2, color = PURPLE_B, fill_opacity=1)
        x_pos = [0, 0.5, 1, 1.5]
        purple_small = [Rectangle(height=0.5, width=0.5, color = PURPLE_E).shift([-0.75 + x, 0, 0]) for x in x_pos]
        query_block = VGroup(purple_big , *purple_small).move_to([0, 0, 0])

        # value
        blue_big = Rectangle(height=0.5, width=2, color = BLUE_B, fill_opacity=1)
        x_pos = [0, 0.5, 1, 1.5]
        blue_small = [Rectangle(height=0.5, width=0.5, color = BLUE_E).shift([-0.75 + x, 0, 0]) for x in x_pos]
        value_block = VGroup(blue_big , *blue_small).move_to([0, -1, 0])

        # ultimate embedding
        gradients = [WHITE, RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE][::-1]
        sizes = [i * 0.05 for i in range(len(gradients))][::-1]
        u_big = [Rectangle(height=0.5 + s, width=2 + s, color = col, fill_opacity=1) for s, col in zip(sizes, gradients)]
        
        x_pos = [0, 0.5, 1, 1.5]
        u_small = [Rectangle(height=0.5, width=0.5, color=BLACK).shift([-0.75 + x, 0, 0]) for x in x_pos]
        u_embed_block = VGroup(*u_big , *u_small).move_to([0, -2, 0])


        ## matrices
        grey_big = Rectangle(height=2, width=2, color = LIGHT_GREY, fill_opacity=1)
        
        # key matrix
        y_pos = [-1.25 + i*0.5 for i in range(4)]
        orange_small_square = [VGroup(*orange_small).copy().shift([0, y, 0]) for y in y_pos]
        key_matrix = VGroup(grey_big.copy().shift([0, 0.50, 0]), *orange_small_square)
        key_matrix.scale(0.5).move_to([0,0,0])

        # value matrix
        y_pos = [-0.25 + i*0.5 for i in range(4)]
        blue_small_square = [VGroup(*blue_small).copy().shift([0, y, 0]) for y in y_pos]
        value_matrix = VGroup(grey_big.copy().shift([0, -0.50, 0]), *blue_small_square)
        value_matrix.scale(0.5).move_to([0,-1,0])



        # attention procedure

        sentence = [word+" " for word in "Never seen movie with such great performances".split(" ")]

        sentence_mob = TextMobject(*sentence).scale(0.6)

        for i in range(len(sentence)):
            if i == 0:
                sentence_mob[i].move_to([-4.5, 1.8, 0])
            elif i == 5:
                sentence_mob[i].next_to(sentence_mob[i-1], RIGHT, buff = 1).shift([0, -0.05, 0])
            else:
                sentence_mob[i].next_to(sentence_mob[i-1], RIGHT, buff = 1).align_to(sentence_mob[i-1], DOWN)


        self.play(Write(sentence_mob))
        self.wait(2.5)

        cbow = TexMobject("C", "B", "O", "W").scale(0.6).move_to([-6, 1.2, 0])
        embeddings_text = TextMobject("Embeddings").scale(0.5).next_to(cbow, DOWN)
        cbow[0].set_color(BLUE)
        cbow[1:4].set_color(ORANGE)
        self.play(GrowFromCenter(cbow), GrowFromCenter(embeddings_text))

        # create embeddings
        embed_pos_x = [word.get_center()[0] for word in sentence_mob]
        coords = [[x, 1, 0] for x in embed_pos_x]
        normal_embeds = [n_embed_block.copy().scale(0.5).move_to(coord) for word, coord in zip(sentence_mob, coords)]

        tmp_anims = [GrowFromCenter(ne) for ne in normal_embeds]
        self.play(AnimationGroup(*tmp_anims), lag_ratio = 0.05)
        self.wait(1)

        ## create keys
        # show key matrix
        self.play(GrowFromCenter(key_matrix))
        self.wait(1)

        # keys
        coords = [[x, -1, 0] for x in embed_pos_x]
        keys = [key_block.copy().scale(0.5).move_to(coord) for word, coord in zip(sentence_mob, coords)]

        key_text = TextMobject("Keys").move_to([-6, -1, 0]).scale(0.7).set_color(GOLD_B)

        tmp_dots = [Dot(key_matrix.get_center()).set_opacity(0) for i in range(len(keys))]

        tmp_anims = [[Transform(normal_embeds[i].copy(), tmp_dots[i]), Transform(tmp_dots[i], keys[i])] for i in range(len(keys))]

        tmp_anims = [item for sublist in tmp_anims for item in sublist]

        self.play(GrowFromCenter(key_text))
        self.play(AnimationGroup(*tmp_anims, lag_ratio = 0.1))
        self.wait(1)

        # fadout query matrix shift queries up
        keys_grouped = VGroup(*tmp_dots)
        self.play(FadeOut(key_matrix))
        self.play(ApplyMethod(keys_grouped.shift, [0, 1, 0]),
                  ApplyMethod(key_text.shift, [0, 1, 0]))
        self.wait(2)


        ## create values
        # show value matrix
        self.play(GrowFromCenter(value_matrix))
        self.wait(1)

        # values
        coords = [[x, -2, 0] for x in embed_pos_x]
        values = [value_block.copy().scale(0.5).move_to(coord) for word, coord in zip(sentence_mob, coords)]
        value_text = TextMobject("Values").move_to([-6, -2, 0]).scale(0.7).set_color(BLUE_B)

        tmp_dots = [Dot(value_matrix.get_center()).set_opacity(0) for i in range(len(values))]

        tmp_anims = [[Transform(normal_embeds[i].copy(), tmp_dots[i]), Transform(tmp_dots[i], values[i])] for i in range(len(values))]

        tmp_anims = [item for sublist in tmp_anims for item in sublist]

        self.play(GrowFromCenter(value_text))
        self.play(AnimationGroup(*tmp_anims, lag_ratio = 0.1))
        self.wait(1)

        # fadout value matrix 
        values_grouped = VGroup(*tmp_dots)
        self.play(FadeOut(value_matrix))
        self.wait(2)


        # create one query
        query_text = TextMobject("Queries").move_to([-6, -1, 0]).scale(0.7).set_color(PURPLE_B)
        coords = [[x, -1, 0] for x in embed_pos_x]

        query = query_block.copy().scale(0.5).move_to(coords[0])

        self.play(GrowFromCenter(query_text), GrowFromCenter(query))
        self.wait(2)

        # match query with keys


            

        self.play(GrowFromCenter(n_embed_block))
        self.wait(2)
        self.play(GrowFromCenter(key_block))
        self.wait(2)
        self.play(GrowFromCenter(query_block))
        self.wait(2)
        self.play(GrowFromCenter(value_block))
        self.wait(2)
        self.play(GrowFromCenter(u_embed_block))
        self.wait(2)

        # short example with query key and value block
        # the query represents what the word is asking for
        # the key is the response
        # so basically a word goes around quering or asking the other words 
        # how important their information is.




        ## B



class SentimentMeasures(Scene):

    def construct(self):
        self.wait(1)
















        