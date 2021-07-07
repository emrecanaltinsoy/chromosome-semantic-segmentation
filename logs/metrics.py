import os
import glob
import argparse

import yaml
import numpy as np

yaml_files = glob.glob("./**/metrics.yaml", recursive=True)
npy_files = glob.glob("./**/test-eval.npy", recursive=True)


def main(args):
    dsc_scores = {}
    evals = {}
    metrics = {}

    if args.print_vals == "dsc_scores":
        print("{:<25}  {:<20}  {:<10}".format("model name", "DSC", "DSC Loss"))
        print("-" * 55)
    elif args.print_vals == "evals":
        print(
            "{:<25}  {:<25}  {:<25}  {:<25}  {:<25}".format(
                "model name", "TP", "TN", "FP", "FN"
            )
        )
    elif args.print_vals == "metrics":
        print(
            "{:<25}  {:<10}  {:<10}  {:<10}  {:<10}  {:<10}".format(
                "model name", "DSC", "SE", "SP", "PRE", "ACC"
            )
        )  #'Ch1 DSC',
        print("-" * 75)

    for i in range(len(yaml_files)):
        net_name = yaml_files[i].split("\\")[-2]
        with open(yaml_files[i]) as f:
            scores = yaml.load(f, Loader=yaml.FullLoader)
            dsc_scores[net_name] = scores
            if args.print_vals == "dsc_scores":
                print(
                    "{:<25}  {:<20}  {:<10}".format(
                        net_name,
                        scores["mean_DSC"] * 100,
                        scores["mean_dsc_loss"] * 100,
                    )
                )

        mask0 = np.zeros(4)
        mask1 = np.zeros(4)
        mask2 = np.zeros(4)
        eval_ = np.load(npy_files[i])
        for e in eval_:
            mask0 += e[0]
            mask1 += e[1]
            mask2 += e[2]

        m0_TP, m0_TN, m0_FP, m0_FN = [float(t) for t in mask0 / 84]
        m1_TP, m1_TN, m1_FP, m1_FN = [float(t) for t in mask1 / 84]
        m2_TP, m2_TN, m2_FP, m2_FN = [float(t) for t in mask2 / 84]
        overall_TP, overall_TN, overall_FP, overall_FN = [
            float(t) for t in (mask0 + mask1 + mask2) / 84
        ]

        coeff = {
            "m0": {"TP": m0_TP, "TN": m0_TN, "FP": m0_FP, "FN": m0_FN},
            "m1": {"TP": m1_TP, "TN": m1_TN, "FP": m1_FP, "FN": m1_FN},
            "m2": {"TP": m2_TP, "TN": m2_TN, "FP": m2_FP, "FN": m2_FN},
            "overall": {
                "TP": overall_TP,
                "TN": overall_TN,
                "FP": overall_FP,
                "FN": overall_FN,
            },
        }

        if args.print_vals == "evals":
            print("-" * 125)
            print(
                "{:<25}  {:<25}  {:<25}  {:<25}  {:<25}".format(
                    net_name,
                    coeff["m0"]["TP"],
                    coeff["m0"]["TN"],
                    coeff["m0"]["FP"],
                    coeff["m0"]["FN"],
                )
            )
            print(
                "{:<25}  {:<25}  {:<25}  {:<25}  {:<25}".format(
                    "",
                    coeff["m1"]["TP"],
                    coeff["m1"]["TN"],
                    coeff["m1"]["FP"],
                    coeff["m1"]["FN"],
                )
            )
            print(
                "{:<25}  {:<25}  {:<25}  {:<25}  {:<25}".format(
                    "",
                    coeff["m2"]["TP"],
                    coeff["m2"]["TN"],
                    coeff["m2"]["FP"],
                    coeff["m2"]["FN"],
                )
            )
            print(
                "{:<25}  {:<25}  {:<25}  {:<25}  {:<25}".format(
                    "",
                    coeff["overall"]["TP"],
                    coeff["overall"]["TN"],
                    coeff["overall"]["FP"],
                    coeff["overall"]["FN"],
                )
            )

        evals[net_name] = coeff

        TP = coeff[args.eval_type]["TP"]
        TN = coeff[args.eval_type]["TN"]
        FP = coeff[args.eval_type]["FP"]
        FN = coeff[args.eval_type]["FN"]

        DSC = (
            (2 * TP) / (FP + FN + (2 * TP)) * 100
            if TP != 0
            else TN / (TN + FP + FN + TP) * 100
        )
        SE = TP / (TP + FN) * 100
        SP = TN / (TN + FP) * 100
        PRE = TP / (TP + FP) * 100
        ACC = (TP + TN) / (TP + TN + FP + FN) * 100

        DSC = round(DSC, 4)
        SE = round(SE, 4)
        SP = round(SP, 4)
        PRE = round(PRE, 4)
        ACC = round(ACC, 4)

        metrics[net_name] = {
            "DSC": DSC,
            "SE": SE,
            "SP": SP,
            "PRE": PRE,
            "ACC": ACC,
        }

        if args.print_vals == "metrics":
            print(
                "{:<25}  {:<10}  {:<10}  {:<10}  {:<10}  {:<10}".format(
                    net_name, DSC, SE, SP, PRE, ACC
                )
            )  # ch1_DSC,

    with open(os.path.join("logs/dsc_scores.yaml"), "w") as fp:
        yaml.dump(dsc_scores, fp)

    with open(os.path.join("logs/evals.yaml"), "w") as fp:
        yaml.dump(evals, fp)

    with open(os.path.join(f"logs/metrics-{args.eval_type}.yaml"), "w") as fp:
        yaml.dump(metrics, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic segmentation of G-banding chromosome Images"
    )
    parser.add_argument(
        "--print-vals",
        type=str,
        default="metrics",
        help="choose which values to print [none, dsc_scores, evals, metrics] (default: metrics)",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        default="overall",
        help="choose which evaluation metrics to print [overall, m0, m1, m2] (default: overall)",
    )
    args = parser.parse_args()
    main(args)
