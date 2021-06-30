import matplotlib.pyplot as plt
import pandas as pd

from graph.plot_graph import PlotGraph
from graph.topology_score import ComponentScore, CycleScore

class PlotGraphMatching:

    def __init__(self, label_path, label_name, pred_path, pred_name):
        self.label_path = label_path
        self.label_name = label_name
        self.pred_path = pred_path
        self.pred_name = pred_name

    def get_label_figure(self):
        return PlotGraph(self.label_path, self.label_name).get_figure()

    def get_prediction_figure(self):
        return PlotGraph(self.pred_path, self.pred_name).get_figure()

    def get_cmp_scores(self):
        cmp_score = ComponentScore(self.label_path, self.label_name, self.pred_path, self.pred_name)
        mean_row_score, mean_column_score = cmp_score.mean_row_and_column_scores()
        final_score = cmp_score.final_score()
        return mean_row_score, mean_column_score, final_score
    
    def get_cyc_scores(self):
        cyc_score = CycleScore(self.label_path, self.label_name, self.pred_path, self.pred_name)
        mean_row_score, mean_column_score = cyc_score.mean_row_and_column_scores()
        final_score = cyc_score.final_score()
        return mean_row_score, mean_column_score, final_score
        
    def make_title(self):
        scores = [list(self.get_cmp_scores()), list(self.get_cyc_scores())]
        return pd.DataFrame(scores, index=['cmp', 'cyc'], columns=['row', 'column', 'final'])

    def get_matching_figure(self):
        plt.figure(figsize=(20,10))
        plt.suptitle(f'{self.make_title()}')
        plt.subplot(121)
        self.get_label_figure()
        plt.subplot(122)
        self.get_prediction_figure()
        return plt.gcf()
    
    def show_figure(self):
        self.get_matching_figure()
        plt.show()

    def save_figure(self):
        self.get_matching_figure()
        plt.savefig(f'{self.pred_path}/{self.pred_name}.png')
        plt.savefig(f'{self.pred_path}/{self.pred_name}.pdf')
        plt.close()

if __name__ == '__main__':
    plot = PlotGraphMatching('alaki', 'seg_LI_2019-08-30_emb2_pos1_tp162_D2D3D4_A2',
                    'alaki', 'one-ten_40_val_pred0.7_LI_2019-08-30_emb2_pos1_tp162_D2D3D4_A2')
    plot.save_figure()
    plot.show_figure()