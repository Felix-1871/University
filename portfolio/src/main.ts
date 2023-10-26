import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './style.css'
import App from './App.vue'
import Skills from './components/Skills.vue'
import Projects from './components/Projects.vue'
import Contact from './components/Contact.vue'
import About from './components/About.vue'
import Home from './components/Home.vue'


import router from './router'
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
const pinia = createPinia()
const vuetify = createVuetify({
    components,
    directives,
})

import { library } from '@fortawesome/fontawesome-svg-core'

/* import font awesome icon component */
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'

/* import specific icons */
import { faSun, faMoon, faBars, faHeart } from '@fortawesome/free-solid-svg-icons'

/* add icons to the library */
library.add(faBars, faSun, faMoon, faHeart)




createApp(App)
.component('font-awesome-icon', FontAwesomeIcon)
.use(router)
.use(vuetify)
.use(pinia)
.mount('#app')
