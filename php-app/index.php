<?php
session_start();
/*
Uncomment, if you want to switch debug mode on

require __DIR__.'/vendor/autoload.php';
$debug = new \bdk\Debug(array(
	'collect' => true,
	'output' => true,
));
$debug->log($_SESSION); */
?>
<!DOCTYPE html>
<html>
  <head>
          <meta charset="utf-8" />
          <title>Kino Bilecik</title>
          <link rel="stylesheet" href="assets/css/style.css" />
<link rel="stylesheet" href="assets/css/fontello/fontello.css" /> 
           <link href="https://fonts.googleapis.com/css?family=Lato:400,900|Raleway:400,900|Shadows+Into+Light&amp;subset=latin-ext" rel="stylesheet" />
           <link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/swiper@8/swiper-bundle.min.css"/>
<script src="https://cdn.jsdelivr.net/npm/swiper@8/swiper-bundle.min.js"></script>


  </head>
  <body>
     <header>
        <h1 class="logo"><i class="icon-video-2"></i> Kino Bilecik</h1>

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
      <div id="zapowiedzi">
          
         
          <!-- Slider main container -->
<div class="swiper">
  <!-- Additional required wrapper -->
  <div class="swiper-wrapper">
    <!-- Slides --> <?php 
            $db = new PDO('sqlite:bazakina.db');
            $movie_query = ($db->query("select * from ZAPOWIEDZI group by TITLE"));


            while ($r = $movie_query->fetch()){
                $cover = $r['cover'];
                
                print '<div class="swiper-slide" style="background-image:url('.$cover.')"> </div>';
                 
                
            }


          ?> 
  </div>


  <!-- If we need navigation buttons -->
  <div class="swiper-button-prev"></div>
  <div class="swiper-button-next"></div>

</div>

 
       </article>
      
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
    <script>
      
      const swiper = new Swiper('.swiper', {
  // Optional parameters
  loop: true,

  // Navigation arrows
  navigation: {
    nextEl: '.swiper-button-next',
    prevEl: '.swiper-button-prev',
  },

  // And if we need scrollbar
  scrollbar: {
    el: '.swiper-scrollbar',
  },
});
</script>
  </body>
  
</html>