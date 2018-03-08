$(document).ready(function() {
  $('#light-slider1, #light-slider2, #light-slider3, #light-slider4').lightSlider({
    item: 1,
    keyPress: false,
    controls: true,
    easing: 'cubic-bezier(0.25, 0, 0.25, 1)',
    rtl:false,
    slideMargin: 100,
    loop: true
  });
});
