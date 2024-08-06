import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D



# Define the groups based on the description provided
groups = {
    "enhancer": ["binary","enhancer_cohn", "enhancer_ensembl"],
    "promoter": ["GM12878", "HUVEC", "Hela-S3", "NHEK", "prom_core_notata", 'prom_core_tata', 
                 'prom_core_all','prom_300_notata', 'prom_300_tata', 'prom_300_all'],
    "splice site": ["donors", "acceptors"],
    "open chromatin": ["ocr", "DNase_I"],
    "TF binding site": ['tf_0', 'tf_1', 'tf_2', 'tf_3', 'tf_4'],
    "coding": ["coding"],
}


def generate_mean(df, groups):
    df = pd.read_csv(df)
    # Calculate the mean for each group
    mean_rows = []
    for group_name, items in groups.items():
        # Filter the rows that match the group items
        group_df = df[df['Data'].isin(items)]
        # Calculate the mean of the numerical columns
        mean_data = group_df[['MCC', 'AUC', 'F1-Score', 'Accuracy']].mean()
        # Add the group name to the mean data
        mean_data['Data'] = group_name
        # Append the result
        mean_rows.append(mean_data)

    # Create a new DataFrame with only the means
    mean_df = pd.DataFrame(mean_rows)

    # Reorder columns to match the original DataFrame structure
    mean_df = mean_df[['Data', 'MCC', 'AUC', 'F1-Score', 'Accuracy']]
    return mean_df

db = generate_mean("dnabert2_ordered.csv", groups)
nt = generate_mean("ntv2_ordered.csv", groups)
hyena = generate_mean("hyena_ordered.csv", groups)

# Create the nested list format
metrics = ['AUC', 'MCC', 'F1-Score', 'Accuracy']
group_names = db['Data'].tolist()
nested_list = [group_names]

for metric in metrics:
    nested_list.append((
        metric,
        [
            db[metric].tolist(),
            nt[metric].tolist(),
            hyena[metric].tolist()
        ]
    ))


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta



if __name__ == '__main__':
    N = 6
    theta = radar_factory(N, frame='polygon')

    data = nested_list
    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(figsize=(9, 8.7), nrows=2, ncols=2,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.6, hspace=0.4, top=0.85, bottom=0.05)

    colors = ['b', 'r', 'g']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        if title == "MCC":
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        else:
            ax.set_rgrids([0.6, 0.7, 0.8, 0.9])

        labels = ax.get_xticklabels()  # Get the labels for the spokes
        for label in labels:
            label.set_position((label.get_position()[0], label.get_position()[1] - 0.06))

        ax.set_title(title, weight='bold', size='x-large', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='bottom')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)
        ax.tick_params(axis='x', labelsize="large")

    # add legend relative to top-left plot
    labels = ('DNABERT-2', "NT-v2", "HyenaDNA")
    legend = axs[0, 0].legend(labels, loc=(0.95, 1),
                              labelspacing=0.1, fontsize='large')

    # fig.text(0.5, 0.92, 'Evaluation on Human Genome CLassification Tasks',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='x-large')

    plt.savefig("./plots/radar.png", dpi=200)