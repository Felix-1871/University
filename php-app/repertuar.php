<?php
session_start();
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
     <div class="seanse">

      <?php

        $db = new PDO('sqlite:bazakina.db');
              
      ?>


      </div>
      <article class="movies">
          <header>
            <h2>Wyświetlane filmy</h2>

          </header>
Aby zarezerwować miejsca na dany seans, kliknij na wybraną godzinę seansu.
          <?php 
            
            $day_array = array(1=>'Poniedziałek', 'Wtorek', 'Środa', 'Czwartek', 'Piątek', 'Sobota', 'Niedziela');
            $movie_query = ($db->query("select * from SEANSE group by TITLE")); //zapytanie o wszystkie filmy, zgrupowane po tytule

            while ($r = $movie_query->fetch()){
                $api_key = '61ff4ada12b1b829d16bc2acf45366d2';
                $title = $r['title'];
                $response = file_get_contents("https://api.themoviedb.org/3/search/movie?api_key=$api_key&query=$title");
                $data = json_decode($response, true);
                $movie = $data['results'][0];
                $cover = $r['cover'];
                print '<section class="movie">';
                print '<figure> <img src="'.$cover.'" width="60% height="60%" alt="plakat"/>';
                print '<figcaption>'.$r['title'].'</figcaption></figure>';
                print '<div class="opis"><p>'.$r['description'].'</br></br>Czas trwania: '.$r['movie_time'].'</p></div>';
                print '<div class="opis2"><h1>Ocena TMDb: '. ''.$movie['vote_average'].'</h1></div>';
                print '<div id="dates">';
                 foreach ($day_array as $day_value => $day_string) {
                    print '<div class="day"><form>'.$day_string.'</b></br>';
                    $day = $day_value;
                    $title = $r['title'];
                    $day_query = ($db->query("select id_seans,time from SEANSE where day = '$day' and title like '$title'"));
                    while ($time = $day_query->fetch()){
                      print '<a href="wybor.php?id_seans='.$time['id_seans'].'" title="Kliknij tutaj zeby zarezerwowac"><input type="button" name="godz" value="'.$time['time'].'"/></a>';
                     }
                     print '</br>';
                print '</form></div>';
                }
            print '</div>';
            print '</section>';
            }

         ?>

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
     
  </body>
  
</html>