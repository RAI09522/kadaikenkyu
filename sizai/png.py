import matplotlib.pyplot as plt

# MatplotlibでLaTeXエンジンを使用するための設定
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# 数式のテキスト
formula = r"$\displaystyle dV_{i,t} = \underbrace{\kappa E_{i,t} e_{i,t}}_{\text{Algorithm}} + \underbrace{\beta_{net} \sqrt{V_{i,t}} E_{i,t}}_{\text{Network}} - \underbrace{\phi V_{i,t}}_{\text{Forget}} + \underbrace{\sigma V_{i,t} dW_{i,t}}_{\text{Shock}}$"

# キャンバスの作成
fig = plt.figure(figsize=(10, 2))
fig.text(0.5, 0.5, formula, fontsize=32, ha='center', va='center')
plt.axis('off')

# 画像として保存（透過PNG）
plt.savefig('visibility_equation_poster.png', dpi=600, bbox_inches='tight', transparent=True)
print("画像を保存しました。")
