import numpy as np
from matplotlib import pyplot as plt, cm

from compare_variance_residual.common_utils.feature_utils import get_prediction_path


language_model = "bert"
feature = "low-level" # semantic low-level joint
modality = "listening"
subject = "01"
layer = 9
# low_level_feature = "numwords"

low_level_features = ["numwords", "letters", "numletters", "word_length_std", "numphonemes", "phonemes"]


correlations = dict()
for low_level_feature in low_level_features:
    # Load the data
    path = get_prediction_path(language_model, feature, modality, subject, low_level_feature, layer)
    correlations[low_level_feature] = np.mean(np.nan_to_num(np.load(path, allow_pickle=True)))

inferno = cm.get_cmap('inferno')
colors = [inferno(0.4)] * 4

# bar plot comparing averages
plt.bar(correlations.keys(), correlations.values(), color=colors)
plt.xticks(rotation=15, ha="right")
plt.ylabel("Average correlation")
plt.xlabel("Low-level feature")
plt.title(f"Average correlation between low-level features and {modality} activations")

plt.savefig(f"../plots/bar_average_correlation_low_level_{language_model}_{feature}_{modality}_{subject}.png")
plt.show()
