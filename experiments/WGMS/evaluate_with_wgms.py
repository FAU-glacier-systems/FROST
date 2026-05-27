import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ref = pd.read_csv("glacier_ELA_gradients.csv")
res = pd.read_csv("../central_europe_submit/tables/aggregated_results.csv")

key = "rgi_id"

ref = ref.rename(columns={
    "ELA_m": "ela",
    "lower_gradient": "gradabl",
    "upper_gradient": "gradacc"
})

df = ref.merge(
    res[[key, "ela", "gradabl", "gradacc"]],
    on=key,
    suffixes=("_ref", "_calib")
)

df.to_csv("aggregated_results_with_wgms.csv", index=False)

# unique glaciers and color map
glaciers = df[key].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(glaciers)))
color_dict = dict(zip(glaciers, colors))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, v in zip(axes, ["ela", "gradabl", "gradacc"]):
    for gid in glaciers:
        sub = df[df[key] == gid]
        ax.scatter(
            sub[f"{v}_ref"],
            sub[f"{v}_calib"],
            color=color_dict[gid],
            label=df['glamos_name']
        )

    # 1:1 line
    xmin = df[f"{v}_ref"].min()
    xmax = df[f"{v}_ref"].max()
    ax.plot([xmin, xmax], [xmin, xmax])

    ax.set_xlabel(f"{v} reference")
    ax.set_ylabel(f"{v} calibrated")
    ax.set_title(v)

# single legend
handles = [
    plt.Line2D([0], [0], marker='o', linestyle='', color=color_dict[gid], label=gid)
    for gid in glaciers
]
fig.legend(handles=handles, title="Glacier", loc="center right")

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()
