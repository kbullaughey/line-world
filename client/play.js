var socket = new WebSocket("ws://localhost:2601", "lineworld");

var command = "stay";
var timer = setInterval(function() {
  socket.send(command);
  command = "stay";
}, 100);

socket.onmessage = function(e) {
  if (e.data.match(/^>/)) {
    $('#scene').html(e.data);
  } else {
    $('#message').html(e.data);
  }
};

socket.onclose = function(e) {
  clearTimeout(timer);
};

$('body').on('keyup', function(e) {
  if (e.keyCode == 74) {
    console.log("left");
    command = "left";
  } else if(e.keyCode == 75) {
    console.log("right");
    command = "right";
  }
});
