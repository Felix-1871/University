<?php
session_start();
$db = new PDO('sqlite:users.db');

if(!isset($_SESSION['user']))
{
  $komunikat = 'Nie byłeś zalogowany.';
}
else {
  unset($_SESSION['priv'], $_SESSION['user']);
  $komunikat = '<p>Zostałeś wylogowana/y.</p>';
  session_destroy();
}

?>

<!DOCTYPE html>
<html>
  <head>
          <meta charset="utf-8" />
          <title>Kino Bilecik</title>
          <link rel="stylesheet" href="assets/css/style.css" />
<link rel="stylesheet" href="assets/css/fontello/fontello.css" />            <link href="https://fonts.googleapis.com/css?family=Lato:400,900|Raleway:400,900|Shadows+Into+Light&amp;subset=latin-ext" rel="stylesheet" />

          </head>
  <body>
     <header>
        <h1 class="logo"
><i class="icon-video-2"></i> Kino Bilecik</h1>

        <nav id="topnav">
          <ul class="menu">
            <li><a href="index.php">Strona główna</a></li>
            <li><a href="repertuar.php">Repertuar</a></li>
            
            
            <li><a href="kontakt.php">Kontakt</a></li>
            <li><a href="<?php if (!isset($_SESSION['user'])) {
              print 'logowanie.php">Zaloguj się</a></li>';
            } elseif (isset($_SESSION['isAdmin'])) {
              print 'panel.php">Panel Administratora</a></li>';
            } else {
              print 'panel.php">Panel Użytkownika</a></li>';} ?>
          </ul>
        </nav>
     </header>
     <div class="content">
     <main>
      <?php

          print $komunikat;
         print '<br /><a href="index.php" class="button">Powrót do strony głównej</a>'.PHP_EOL;
         print '<br /><a href="logowanie.php" class="button">Zaloguj ponownie</a>'.PHP_EOL;
         
      ?>

  
     </main>
     </div>
     
    <footer>
      <div class="socials">
            <div class="fb">
              <i class="icon-facebook-official"></i>
            </div>
            <div class="tw">
              <i class="icon-twitter"></i>
            </div>
            <div class="insta">
              <i class="icon-instagram"></i>
            </div>            
      </div> 

      <div class="info">
Zapraszamy na seanse w naszym kinie!
      </div>

    </footer>
  </body>
  
</html>
