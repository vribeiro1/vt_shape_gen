<h1>Vocal Tract Shape Generator</h1>

<p>
    Automatic generation of the complete vocal tract shape from the sequence of phonemes to be articulated.
</p>

<h2>How to contribute?</h2>

<ol>

<li>Clone the repository</li>

```
>> git clone git@gitlab.inria.fr:vsouzari/vt_shape_gen.git
```

<li> Create and activate your virtual environment</li>

```
>> cd vt_shape_gen
>> python3 -m venv .dev_env
>> source .dev_env/bin/activate
```

<li>Clone vt_tools</li>

```
>> cd ..
>> git clone git@gitlab.inria.fr:vsouzari/vt_tools.git
>> cd vt_shape_gen
```

<li>Install vt_tools </li>

```
>> pip3 install -e ../vt_tools
```

<li>Install the requirements </li>

```
>> pip3 install -r requirements.txt
```

<li>Run the tests</li>

```
>> py.test test
```

</ol>