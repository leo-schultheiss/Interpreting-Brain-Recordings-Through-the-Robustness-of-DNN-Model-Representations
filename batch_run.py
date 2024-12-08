import itertools

from compare_variance_residual import low_level_prediction, semantic_prediction, joint_prediction

def start_batch_run():
    print("Batch run started")

    for subject in range(1, 10):
        for modality in ["reading", "listening"]:
            for low_level_feature in ["letters", "numletters", "numphonemes", "numwords", "phonemes", "word_length_std"]:
                low_level_prediction.train_low_level_model("./data", subject, modality, low_level_feature)

            for layer in range(1, 13):
                for feature in ["semantic"]:
                    semantic_prediction.predict_brain_activity("data", "bert_base20.npy", "bert", layer, subject, modality,  feature)

                # join all possible pairs of textual features, e.g "semantic,letters", "semantic,numletters", etc.
                for feature_1, feature_2 in itertools.product(["semantic", "letters", "numletters", "numphonemes", "numwords", "phonemes", "word_length_std"], repeat=2):
                    if feature_1 == feature_2:
                        continue
                    joint_prediction.predict_joint_model("data", "english1000sm.hf5", "bert", subject, modality, layer, ",".join([feature_1, feature_2]))

    print("Batch run completed")


if __name__ == "__main__":
    start_batch_run()