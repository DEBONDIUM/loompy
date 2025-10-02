# Loompy
# 🧵 Loompy – Mesh Generation for Weave Simulations

**Loompy** is a lightweight Python tool designed to **generate meshes tailored for weaving and textile simulations**. Whether you're studying yarn interactions, fabric mechanics, or complex woven patterns, `loompy` helps you build consistent meshes that capture the geometry of interlacing threads.  

> 🪡 From yarn to mesh — fast, clear, and **woven with care**.

---

## 📦 Features

- 🧵 Generate structured meshes for woven textiles  
- 🧮 Define custom weave patterns (plain, twill, satin, or user-defined)  
- 🔗 Automatic handling of thread interlacing and contact points  
- 📐 Export meshes to standard formats for simulation tools  
- 🧰 Lightweight and easy to integrate with existing FEM/DEM workflows  

---

## 🚀 Getting Started

### Installation

```bash
pip install git+https://github.com/DEBONDIUM/loompy.git
```

You can also clone and install manually:

```bash
git clone https://github.com/DEBONDIUM/loompy.git
cd crispy
pip install .
```

---

## 🧪 Example Usage

```python
import loompy as lp

# Define a simple plain weave with 2 warp and 2 weft yarns
# considering segments defined before
m = lp.Mesh(warp = [warp_segment_010, warp_segment_101], 
            weft = [weft_segment_101, weft_segment_010], 
            space = 1, 
            num_warp = 2, num_weft = 2, 
            num_segment_warp = 1, num_segment_weft = 1)

# Visualize
m.plot()

# Export to mesh format

m.export('msh')
```

For full working examples and more examples available, see the [`examples/`](examples/) folder and in the [documentation](https://debondium.github.io/loompy).

---

## 📘 Documentation

The full documentation (installation, API reference, usage examples) is available at:  
🔗 https://debondium.github.io/loompy

---

## 🔬 Applications

- Textile mechanics  
- Weaving process simulation
- Yarn-level finite element modeling
- Multiscale fabric analysis
- Contact/friction studies in woven structures

---

## 🛠 Dependencies

- [Numpy](https://numpy.org)  
- [PyVista](https://pyvista.org/)  

---

## 🧑‍💻 Contributing

Pull requests are welcome!  
Feel free to fork the repository, propose new features, or report bugs via GitHub Issues.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍🏫 Acknowledgments

This library is developed for researchers and engineers working in **textile mechanics**, **multiscale modeling**, and **computational weaving**.  
If you use `loompy` in your work, please consider citing it or linking to the repository.

> _“In the breaking of things lies the story of how they were made.”_
