
<!DOCTYPE html>
<html lang="en">
<head>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
  <link rel="stylesheet" href="assets/css/panel.css">
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>

</head>
<body>
   
  <input type="checkbox" id="check">

  <label for="check">
<i class="fa fa-bars" id="btn"></i>
<i class="fa fa-times" id="cancle"></i>


  </label>
  <div class="sidebar">
<header>
<img src="assets/images/i don't wanna do this anymore/profile-pic.png">
<p>Administrator</p>
</header>
<ul>
    <li><a href="#"><i class="fa fa-qrcode"></i>Dashboard</a></li>
    <li><a href="404.php"><i class="fa fa-link"></i>Shortcuts</a></li>
    <li><a href="404.php"><i class="fa fa-eye"></i>Overview</a></li>
    <li><a href="edytrezerwacje.php"><i class="fa fa-book"></i>Rezerwacje</a></li>
    <li><a href="dodajseans.php"><i class="fa fa-eye"></i>Seanse</a></li>
    <li><a href="404.php"><i class="fas fa-cog"></i>Service</a></li>
    <li><a href="wyloguj.php"><i class="fas fa-user-shield"></i>Wyloguj się</a></li>
    </ul>
   <li>
      <div class="social-links">
        <a href="#"><i class="fab fa-facebook"></i></a>
        <a href="#"><i class="fab fa-youtube"></i></a>
        <a href="#"><i class="fab fa-twitter"></i></a>
        <a href="#"><i class="fab fa-whatsapp"></i></a>
        
      </div>
    </li>
   

</div> 
<h1>Wybierz aplikację:</h1>
<div class="wrapper">
 
<div id="grid-col">
  <div class="cell">
<h3>Rezerwacje:</h3>
<center><a href="edytrezerwacje.php"><button>Edytuj</button></a> </center>
  </div> 
  <div class="cell">
<h3>Seanse</h3>
<center> <a href="dodajseans.php"><button>Dodaj</button></a> </center>
  </div>
  <div class="cell">
<h3>Wyloguj się</h3>
<center> <a href="wyloguj.php"><button>Wyloguj się</button></a> </center>
  </div>
</div>
</div>
<div style="height: 80vmin;"></div>
<footer>
</footer>
</body>
</html>