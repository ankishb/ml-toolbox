import heapq
 
class Beam(object):
#For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
#This is so that if two prefixes have equal probabilities then a complete sentence is preferred over an incomplete one since (0.5, False) < (0.5, True)
 
    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width
 
    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)


def beamsearch(probabilities_function, beam_width=10, clip_len=-1):
    prev_beam = Beam(beam_width)
    prev_beam.add(1.0, False, [ '<start>' ])
    while True:
        curr_beam = Beam(beam_width)

        #Add complete sentences that do not yet have the best probability to the current beam, the rest prepare to add more words to them.
        for (prefix_prob, complete, prefix) in prev_beam:
            if complete == True:
                curr_beam.add(prefix_prob, True, prefix)
            else:
                #Get probability of each possible next word for the incomplete prefix.
                for (next_prob, next_word) in probabilities_function(prefix):
                    if next_word == '<end>': #if next word is the end token then mark prefix as complete and leave out the end token
                        curr_beam.add(prefix_prob*next_prob, True, prefix)
                    else: #if next word is a non-end token then mark prefix as incomplete
                        curr_beam.add(prefix_prob*next_prob, False, prefix+[next_word])

        (best_prob, best_complete, best_prefix) = max(curr_beam)
        if best_complete == True or len(best_prefix)-1 == clip_len: #if most probable prefix is a complete sentence or has a length that exceeds the clip length (ignoring the start token) then return it
            return (best_prefix[1:], best_prob) #return best sentence without the start token and together with its probability

        prev_beam = curr_beam

