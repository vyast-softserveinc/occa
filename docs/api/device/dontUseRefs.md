
<h1 id="dont-use-refs">
 <a href="#/api/device/dontUseRefs" class="anchor">
   <span>dontUseRefs</span>
  </a>
</h1>

<div class="signature">
  <hr>

  
  <div class="definition-container">
    <div class="definition">
      <code>void occa::device::dontUseRefs()</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/26e3076e/include/occa/core/device.hpp#L185" target="_blank">Source</a>
    </div>
    
  </div>


  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/dontUseRefs?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

By default, a [device] will automatically call [free](/api/device/free) through reference counting.
Turn off automatic garbage collection through this method.