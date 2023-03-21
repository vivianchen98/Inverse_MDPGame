### A Pluto.jl notebook ###
# v0.19.17

using Markdown
using InteractiveUtils

# ╔═╡ f8b339a4-abec-11ed-0f33-bf9c3e644669
begin
	using PlutoUI
	using Markdown
	using InteractiveUtils
	TableOfContents(title="Inverse Learning in MDP Games", depth=4)
end

# ╔═╡ d7d2c55a-69e7-4829-9acb-04bb94362012
begin
	using Dates
	md"**Shenghui Chen**, last updated $(now())"
end

# ╔═╡ cfe40d3b-e5be-45c8-90a9-ec125f1613e4
md"""
# Inverse Learning in MDP Games
"""

# ╔═╡ 8ab1feb0-b483-42c8-b6a9-8ac9466cd66b
md"""
## 1. Infinite-horizon MDP

### 1.1 Standard Definition

A infinite-horizon Markov decision process (MDP) is a tuple below:

$(\mathcal{S}, \mathcal{A}, \mathcal{T}, q, c)$

Variable | Description | Definition
:----------- | :-------------- | :------:
$\mathcal{S}$ | State space | $\mathcal{S}=\{S_t\}_{t=1}^T, S_t\in[n]$ where $n\in\mathbb{N}$
$\mathcal{A}$ | Action space | $\mathcal{A}=\{A_t\}_{t=1}^T, A_t\in[m]$ where $m\in \mathbb{N}$
$\mathcal{T}$ | Transition function | $\mathcal{T}_{s'sa}:=\mathbb{P}(S_{t+1}=s'\|S_t=s, A_t=a) \quad \forall t\in[T-1]$
$q$ | Initial State Dist | $q_s:=\mathbb{P}(S_1=s)$
$R$ | Reward function | $R:= S_t\times A_t\to \mathbb{R} \quad \forall t\in[T]$
"""

# ╔═╡ 74519cee-f522-49c4-993f-d1885188b25b
md"""
!!! danger ""
	remember to adapt to infinite horizon!

A policy $\pi\in \mathbb{R}^{T\times n\times m}$ is the way an agent should act in an MDP such that

$(\pi)_{tsa}:=\mathbb{P}(A_t=a|S_t=s) \quad \forall t\in[T], s\in[n], a\in[m]$

An optimal policy attempts to minimize the total expected cost while satisfying the constraints in the MDP (initial state dist, transition function, and that it is in a probability simplex):

!!! info ""
	$\begin{align}
	\min_{\{\mathbb{P}(S_t,A_t)\}_{t=1}^T} \quad &\sum_{t=1}^T \mathbb{E}[c(S_t,A_t)] \tag{1.1a}\\
	\text{s.t.} \quad &\sum_{a}\mathbb{P}(S_1=s, A_1=a)=q_s \tag{1.1b}\\
	& \sum_a\mathbb{P}(S_{t+1},A_{t+1}=a)=\sum_{s,a}\mathcal{T}_{s'sa}\cdot \mathbb{P}(S_t,A_t) \quad \forall t\in[T-1] \tag{1.1c}\\
	& \mathbb{P}(S_t,A_t) \ge 0 \quad \forall t\in[T] \tag{1.1d}
	\end{align}$
"""

# ╔═╡ 1bc19080-f8cb-47b3-91e7-35b9ce686b21
md"""
### 1.2 Linear Program
"""

# ╔═╡ e8d02de1-a900-41c3-9a33-666c5e5ab965
md"""
## 2. Finite-horizon MDP Games
"""

# ╔═╡ 8c87cfe4-60a0-4899-979a-d2b0b88ab578
md"""
### 2.1 Standard Definition

We introduce a multiplayer infinite-horizon MDP game where each player searches for its optimal policy. We denote the total number of players in the game as $p\in\mathbb{N}$.

For each player $i\in p$, we have a tuple

$(\mathcal{S}^i, \mathcal{A}^i, \mathcal{T}^i, q^i, c^i)$

Variable | Description | Definition
:----------- | :-------------- | :------:
$\mathcal{S}^i$ | State space | $\mathcal{S^i}=\{S_t^i\}_{t=1}^T, S_t^i\in[n^i]$ where $n\in\mathbb{N}$
$\mathcal{A}^i$ | Action space | $\mathcal{A}^i=\{A_t^i\}_{t=1}^T, A_t^i\in[m^i]$ where $m\in \mathbb{N}$
$\mathcal{T}^i$ | Transition function | $\mathcal{T}^i_{s'sa}:=\mathbb{P}(S_{t+1}^i=s'\|S_t^i=s, A_t^i=a) \quad \forall t\in[T-1]$
$q^i$ | Initial State Dist | $q_s^i:=\mathbb{P}(S_1^i=s)$
$R^i$ | Cost function | $R^i:= S_t^i\times A_t^1 \times \dots \times A_t^p \to \mathbb{R} \quad \forall t\in[T]$

"""

# ╔═╡ a9530a9d-50fa-437d-8866-9d9e774cd483
md"""
For each player $i\in[p]$, a policy $\pi^i\in \mathbb{R}^{T\times n^i\times m^i}$ is the way it should act in its MDP such that

$(\pi^i)_{tsa}:=\mathbb{P}(A_t^i=a|S_t^i=s) \quad \forall t\in[T], s\in[n^i], a\in[m^i]$

Each player's optimal policy attempts to minimize the total expected cost while satisfying the constraints in the MDP (initial state dist, transition function, and that it is in a probability simplex):

!!! warning ""
	$\begin{align}
	\min_{\{\mathbb{P}(S_t^i,A_t^i)\}_{t=1}^T} \quad &\sum_{t=1}^T \mathbb{E}[c_t^i(S_t,A_t)] \tag{2.1a}\\
	\text{s.t.} \quad &\sum_{a}\mathbb{P}(S_1^i=s, A_1^i=a)=q_s^i \tag{2.1b}\\
	& \sum_a\mathbb{P}(S_{t+1}^i,A_{t+1}^i=a)=\sum_{s,a}\mathcal{T}_{s'sa}^i\cdot \mathbb{P}(S_t^i,A_t^i) \quad \forall t\in[T-1] \tag{2.1c}\\
	& \mathbb{P}(S_t^i,A_t^i) \ge 0 \quad \forall t\in[T] \tag{2.1d}
	\end{align}$
"""

# ╔═╡ f0e2f894-8be7-4e27-ac13-0bda8bf40713
md"""
### 2.2 Nonlinear Program

decision variable: 

$\begin{align}
Y^i_{sa} &:= \sum_{t=0}^\infty \gamma^t \mathbb{P}(S_t^i=s, A_t^i=a) \quad \forall s,a
\end{align}$
"""

# ╔═╡ 6a840201-9522-4eb2-b1ee-1f9b20873832
md"""
### 2.3 Causal Entropy

We need to solve the coupled optimization problem

$\forall i \in [p] \quad \begin{cases}
\begin{align}
\max_{Y^i\in\mathbb{R}^{n^i\times m^i}} \quad & vec(Y^i)^\top \left(b^i+\frac{1}{2}C^{ii}vec(Y^i)+\sum_{j\ne i} C^{ij}vec(Y^j)\right) \\
 &- \sum_{s,a} Y^i_{sa}\left[\log(Y^i_{sa}) - \log \left(\sum_{a'} Y^i_{sa'}\right)\right] \tag{4.1a}\\

\text{s.t.} \quad & D^i vec(Y^i) = q^i + \gamma E^i vec(Y^i) \tag{4.1b}\\

\end{align}
\end{cases}$
"""

# ╔═╡ c866a5f4-a1fe-4533-9ffc-406e5ddd12ae
md"""
### 2.4 Least Squares Problem

Lagrangian

$\begin{align}
\mathcal{L}(Y^i, v^i) &= - vec(Y^i)^\top \left(b^i+\frac{1}{2}C^{ii}vec(Y^i)+\sum_{j\ne i} C^{ij}vec(Y^j)\right) \\
 &+ \sum_{s,a} Y^i_{sa}\left[\log(Y^i_{sa}) - \log \left(\sum_{a'} Y^i_{sa'}\right)\right]\\

&+ (v^i)^\top \left(D^i vec(Y^i) - q^i - \gamma E^i vec(Y^i)\right) \tag{4.2}
\end{align}$
"""

# ╔═╡ a67a1da1-b76b-4574-831a-c8c9b2dddc50
md"""
Stationary conditions

$\begin{align}
\nabla_{vec(Y^i)} \mathcal{L} = -b^i - \sum_{j}C^{ij} vec(Y^j)) + \log(vec(Y^i)) - \log\left((D^i)^\top D^i vec(Y^i)\right) \\
+ (D^i-\gamma E^i)^\top v^i = \mathbf{0}_{n^im^i}
\end{align}$
"""

# ╔═╡ fa897fdf-8656-46a0-802d-00caef213fa4
md"""
Then we have KKT conditions $\forall i\in[p]$:

$\begin{align}
	&-b^i - \sum_{j}C^{ij} vec(Y^j) + \log(vec(Y^i)) - \log\left((D^i)^\top D^i vec(Y^i)\right) + (D^i-\gamma E^i)^\top v^i = 0_{n^im^i} \tag{4.3a}\\

	&D^i vec(Y^i) - q^i - \gamma E^i vec(Y^i) = 0_{n^i} \tag{4.3b}\\

\end{align}$

where
* (a) is the stationarity condition
* (b) is the primal feasibility condition
"""

# ╔═╡ 8577c38b-9d89-4940-9786-dedaa773e5d9
md"""
Let $y = \begin{bmatrix}vec(Y^1), vec(Y^2), \dots, vec(Y^p)\end{bmatrix}$, we can rewrite the left-hand-side equations:

$\begin{align}
	&-b^i - \sum_{j}C^{ij} y^j + \log(y^i) - \log\left((D^i)^\top D^i y^i\right)+ (D^i-\gamma E^i)^\top v^i = 0_{n^im^i} \tag{4.3a}\\

	&D^i y^i - q^i - \gamma E^i y^i = 0_{n^i} \tag{4.3b}\\

\end{align}$
"""

# ╔═╡ cb08e3bd-49dc-42fe-a410-bb05b24a0a3e
md"""
Nonlinear least squares problem:

!!! tip ""
	$\begin{align}	
	\min_{y\in\mathbb{R}^{n^im^i\times p}, v\in\mathbb{R}^{n^i\times p}} \quad \sum_{i=1}^p \left( \|(4.3a)\|^2 + \|(4.3b)\|^2 \right) \tag{2.6}
	\end{align}$
"""

# ╔═╡ 8fac5156-b456-4346-bab6-1a548053b6d2
md"""
## Appendix
"""

# ╔═╡ 56faed82-9ca8-49d3-8763-8c11aa22f800
md"""
### A. Causal Entropy Derivative

Recall we have causal entropy
$\begin{align}
 H(A_{1:T}^i\| S_{1:T}^i) = \sum_{t=1}^T (y^i_t)^{\top} (-\log \pi^i_t) = \sum_{t=1}^T \sum_{s\in[n^i],a\in[m^i]} (y_t^i)_{tsa} \cdot [-\log \pi_{tsa}^i]\\
\end{align}$

Now we want to derive its derivative with respect to $y^i_{tsa}$, notice for a specific timestep $t$, we only need to consider

$\begin{align}
 \nabla_{y^i_{tsa}} H(A_{1:T}^i\| S_{1:T}^i) &= \nabla_{y^i_{tsa}} \left[\sum_{t=1}^T \sum_{s\in[n^i],a\in[m^i]} (y_t^i)_{tsa} \cdot [-\log \pi_{tsa}^i]\right]\\

 &= \nabla_{y^i_{tsa}} \left[\sum_{s\in[n^i],a\in[m^i]} y^i_{tsa} (-\log \pi^i_{tsa})\right]\\

 &= \nabla_{y^i_{tsa}} \left[\sum_{a\in[m^i]} y^i_{tsa} (-\log \pi^i_{tsa})\right]\\

 &= \nabla_{y^i_{tsa}}  \left[y^i_{tsa}(-\log \pi^i_{tsa}) + \sum_{a' \ne a} y^i_{tsa'} (-\log \pi^i_{tsa'})\right]\\

 &= \left(-\log \pi^i_{tsa} + y^i_{tsa} \nabla_{y^i_{tsa}} (-\log \pi^i_{tsa})\right) + \sum_{a'\ne a} y^i_{tsa'} \nabla_{y^i_{tsa}} \left(- \log \pi^i_{tsa'}\right)\\

 &= \left(-\log \pi^i_{tsa} + y^i_{tsa} \nabla_{y^i_{tsa}} (-\log y^i_{tsa} + \log \sum_{a} y^i_{tsa})\right) + \sum_{a'\ne a} y^i_{tsa'} \nabla_{y^i_{tsa}} \left(-\log y^i_{tsa'} + \log (\sum_{a} y^i_{tsa})\right)\\


 &= -\log \pi^i_{tsa} + y^i_{tsa} \left(\frac{1}{\sum_a y^i_{tsa}} - \frac{1}{y^i_{tsa}}\right) + \sum_{a'\ne a} y^i_{tsa'} \left(\frac{1}{\sum_a y^i_{tsa}}\right)\\

 &= -\log \pi^i_{tsa} + \pi^i_{tsa} - 1 + \sum_{a'\ne a} \pi^i_{tsa'}\\

 &= \sum_{a\in[m^i]} \pi^i_{tsa} - \log\pi^i_{tsa}-1
\end{align}$

In this project, we assume policies are defined to have $\sum_a \pi^i_{tsa} = 1$, thus we have

$\begin{align}
\nabla_{y^i_{tsa}} H(A_{1:T}^i\| S_{1:T}^i) = - \log \pi^i_{tsa}
\end{align}$

In vector form, we have

$\nabla_{y^i_t} H(A_{1:T}^i\| S_{1:T}^i) = \begin{bmatrix}\vdots \\ - \log \pi^i_{tsa} \\ \vdots\end{bmatrix}_{s\in[n^i], a\in[m^i]} 
= - \log \pi^i_t$
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
InteractiveUtils = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.49"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.1"
manifest_format = "2.0"
project_hash = "78bb652a8711ca1ff1847c7d1afb641fe254aeb0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "946b56b2135c6c10bbb93efad8a78b699b6383ab"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.6"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─cfe40d3b-e5be-45c8-90a9-ec125f1613e4
# ╟─d7d2c55a-69e7-4829-9acb-04bb94362012
# ╟─8ab1feb0-b483-42c8-b6a9-8ac9466cd66b
# ╟─74519cee-f522-49c4-993f-d1885188b25b
# ╟─1bc19080-f8cb-47b3-91e7-35b9ce686b21
# ╟─e8d02de1-a900-41c3-9a33-666c5e5ab965
# ╟─8c87cfe4-60a0-4899-979a-d2b0b88ab578
# ╟─a9530a9d-50fa-437d-8866-9d9e774cd483
# ╟─f0e2f894-8be7-4e27-ac13-0bda8bf40713
# ╟─6a840201-9522-4eb2-b1ee-1f9b20873832
# ╟─c866a5f4-a1fe-4533-9ffc-406e5ddd12ae
# ╟─a67a1da1-b76b-4574-831a-c8c9b2dddc50
# ╟─fa897fdf-8656-46a0-802d-00caef213fa4
# ╟─8577c38b-9d89-4940-9786-dedaa773e5d9
# ╟─cb08e3bd-49dc-42fe-a410-bb05b24a0a3e
# ╟─8fac5156-b456-4346-bab6-1a548053b6d2
# ╟─56faed82-9ca8-49d3-8763-8c11aa22f800
# ╠═f8b339a4-abec-11ed-0f33-bf9c3e644669
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
