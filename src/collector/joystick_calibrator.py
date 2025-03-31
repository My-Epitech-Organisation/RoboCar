"""
Module pour la calibration du joystick.

Ce module fournit une interface graphique pour calibrer le joystick
et enregistrer les valeurs maximales/minimales pour chaque axe.
"""

import pygame
import time
import math
from utils_collector import joystick_calibration


class JoystickCalibratorUI:
    """Interface graphique pour calibrer le joystick."""

    def __init__(self, joystick):
        """
        Initialise l'interface de calibration.

        Args:
            joystick: L'instance de joystick Pygame à calibrer
        """
        self.joystick = joystick
        if not joystick:
            print("[ERREUR] Aucun joystick détecté pour la calibration")
            return

        # Initialisation de Pygame
        if not pygame.get_init():
            pygame.init()

        # Création de la fenêtre
        self.width, self.height = 800, 600
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Calibration du Joystick")

        # Couleurs modernes
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (100, 100, 100)
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_GRAY = (50, 50, 50)

        # Couleurs vives
        self.RED = (255, 59, 48)
        self.GREEN = (52, 199, 89)
        self.BLUE = (0, 122, 255)
        self.ORANGE = (255, 149, 0)
        self.PURPLE = (175, 82, 222)
        self.TEAL = (48, 176, 199)

        # Couleurs de fond
        self.BACKGROUND_TOP = (22, 23, 30)
        self.BACKGROUND_BOTTOM = (43, 45, 66)

        # Polices améliorées
        try:
            # Essayer de charger une police système plus élégante
            self.font = pygame.font.SysFont("Arial", 36, bold=True)
            self.small_font = pygame.font.SysFont("Arial", 24)
            self.tiny_font = pygame.font.SysFont("Arial", 18)
        except:
            # Fallback sur la police par défaut
            self.font = pygame.font.SysFont(None, 36)
            self.small_font = pygame.font.SysFont(None, 24)
            self.tiny_font = pygame.font.SysFont(None, 18)

        # Variables pour la calibration
        self.steering_min = 0.0
        self.steering_max = 0.0
        self.accel_min = 0.0
        self.accel_max = 0.0

        # État de la calibration
        self.is_calibrating = True
        self.start_time = time.time()
        self.calibration_duration = 10  # Durée en secondes

        # Variables pour les animations
        self.pulse_value = 0
        self.pulse_direction = 1

    def update_calibration_values(self):
        """Met à jour les valeurs min/max selon les entrées actuelles."""
        if not self.joystick:
            return

        # Lecture des valeurs actuelles
        steering_value = self.joystick.get_axis(0)
        accel_value = self.joystick.get_axis(1)

        # Mise à jour des valeurs min/max pour la direction
        if steering_value < self.steering_min:
            self.steering_min = steering_value
        if steering_value > self.steering_max:
            self.steering_max = steering_value

        # Mise à jour des valeurs min/max pour l'accélération
        if accel_value < self.accel_min:
            self.accel_min = accel_value
        if accel_value > self.accel_max:
            self.accel_max = accel_value

        # Mise à jour de l'effet de pulsation
        self.pulse_value += 0.1 * self.pulse_direction
        if self.pulse_value >= 1.0:
            self.pulse_direction = -1
        elif self.pulse_value <= 0.0:
            self.pulse_direction = 1

    def save_calibration(self):
        """Enregistre les valeurs de calibration."""
        # Vérification des valeurs
        if self.steering_min == self.steering_max:
            self.steering_min = -1.0
            self.steering_max = 1.0

        if self.accel_min == self.accel_max:
            self.accel_min = -1.0
            self.accel_max = 1.0

        # Mise à jour et sauvegarde
        joystick_calibration.calib_data["steering"]["min"] = self.steering_min
        joystick_calibration.calib_data["steering"]["max"] = self.steering_max
        joystick_calibration.calib_data["acceleration"]["min"] = self.accel_min
        joystick_calibration.calib_data["acceleration"]["max"] = self.accel_max

        joystick_calibration.save_calibration()

    def draw_gradient_background(self):
        """Dessine un fond dégradé."""
        for y in range(self.height):
            # Interpolation linéaire entre les couleurs du haut et du bas
            ratio = y / self.height
            r = int(self.BACKGROUND_TOP[0] * (1 - ratio) + self.BACKGROUND_BOTTOM[0] * ratio)
            g = int(self.BACKGROUND_TOP[1] * (1 - ratio) + self.BACKGROUND_BOTTOM[1] * ratio)
            b = int(self.BACKGROUND_TOP[2] * (1 - ratio) + self.BACKGROUND_BOTTOM[2] * ratio)
            pygame.draw.line(self.window, (r, g, b), (0, y), (self.width, y))

    def draw_text_with_shadow(self, text, font, color, position, shadow_offset=2):
        """Dessine du texte avec une ombre portée."""
        shadow = font.render(text, True, self.DARK_GRAY)
        text_surface = font.render(text, True, color)
        self.window.blit(shadow, (position[0] + shadow_offset, position[1] + shadow_offset))
        self.window.blit(text_surface, position)

    def draw_rounded_rect(self, surface, rect, color, radius=15):
        """Dessine un rectangle aux coins arrondis."""
        rect = pygame.Rect(rect)
        color = pygame.Color(*color)

        # Dessiner 4 coins arrondis
        pygame.draw.circle(surface, color, (rect.left + radius, rect.top + radius), radius)
        pygame.draw.circle(surface, color, (rect.right - radius - 1, rect.top + radius), radius)
        pygame.draw.circle(surface, color, (rect.left + radius, rect.bottom - radius - 1), radius)
        pygame.draw.circle(surface, color, (rect.right - radius - 1, rect.bottom - radius - 1), radius)

        # Dessiner 4 rectangles pour remplir
        pygame.draw.rect(surface, color, (rect.left + radius, rect.top, rect.width - 2 * radius, rect.height))
        pygame.draw.rect(surface, color, (rect.left, rect.top + radius, rect.width, rect.height - 2 * radius))

    def draw_progress_bar(self, x, y, width, height, progress, bg_color, fill_color, border_color):
        """Dessine une barre de progression avec un joli style."""
        # Fond de la barre
        self.draw_rounded_rect(self.window, (x, y, width, height), bg_color)

        # Calcul de la largeur de remplissage
        fill_width = int(width * progress)

        # Remplissage (s'assurer qu'il ne dépasse pas la largeur totale)
        if fill_width > 0:
            if fill_width < 2 * height:  # Si le remplissage est trop petit pour les coins arrondis
                pygame.draw.rect(self.window, fill_color, (x, y, fill_width, height))
            else:
                self.draw_rounded_rect(self.window, (x, y, fill_width, height), fill_color)

        # Bordure
        pygame.draw.rect(self.window, border_color, (x, y, width, height), 2, border_radius=height//2)

    def draw_axis_visualization(self, x, y, width, height, value, min_val, max_val, label, color):
        """Dessine la visualisation d'un axe avec un style moderne."""
        # Fond avec coins arrondis
        self.draw_rounded_rect(self.window, (x, y, width, height), self.DARK_GRAY)

        # Position actuelle
        normalized = 0.5
        if min_val != max_val:
            if value >= 0:
                normalized = 0.5 + (value / max_val) * 0.5 if max_val != 0 else 0.5
            else:
                normalized = 0.5 - (value / min_val) * 0.5 if min_val != 0 else 0.5

        pos_x = x + int(width * normalized)

        # Curseur avec effet pulsatoire
        pulse_size = 5 + int(2 * self.pulse_value)
        pygame.draw.circle(self.window, color, (pos_x, y + height // 2), pulse_size)
        pygame.draw.circle(self.window, self.WHITE, (pos_x, y + height // 2), pulse_size - 2)

        # Trait vertical à la position
        pygame.draw.line(self.window, color, (pos_x, y), (pos_x, y + height), 2)

        # Texte avec ombre
        text = f"{label}: {value:.2f} (Min: {min_val:.2f}, Max: {max_val:.2f})"
        self.draw_text_with_shadow(text, self.small_font, self.WHITE, (x, y - 30))

        # Marques pour 0
        zero_x = x + width // 2
        pygame.draw.line(self.window, self.GREEN, (zero_x, y), (zero_x, y + height), 2)

        # Ajouter des marques pour les extrêmes
        mark_min = x + width // 4
        mark_max = x + 3 * (width // 4)
        for mark_x in [mark_min, mark_max]:
            pygame.draw.line(self.window, self.LIGHT_GRAY, (mark_x, y), (mark_x, y + height), 1)

            # Petites graduations
            for i in range(1, 10):
                grad_x = mark_min + (mark_max - mark_min) * i / 10
                grad_height = height // 4 if i % 5 == 0 else height // 8
                pygame.draw.line(
                    self.window,
                    self.GRAY,
                    (grad_x, y + height - grad_height),
                    (grad_x, y + height),
                    1
                )

    def draw_joystick_position(self, x, y, size):
        """Dessine une représentation visuelle améliorée de la position du joystick."""
        if not self.joystick:
            return

        # Cercle extérieur (bordure)
        pygame.draw.circle(self.window, self.LIGHT_GRAY, (x, y), size//2, 2)

        # Cercle intérieur (zone de deadzone)
        deadzone_size = size//10
        pygame.draw.circle(self.window, self.DARK_GRAY, (x, y), deadzone_size)

        # Lignes de quadrillage
        pygame.draw.line(self.window, self.GRAY, (x - size//2, y), (x + size//2, y), 1)
        pygame.draw.line(self.window, self.GRAY, (x, y - size//2), (x, y + size//2), 1)

        # Ajouter des cercles concentriques
        for r in range(1, 4):
            pygame.draw.circle(
                self.window,
                self.GRAY,
                (x, y),
                (size//2) * r // 4,
                1
            )

        # Position actuelle du joystick
        steering = self.joystick.get_axis(0)
        accel = -self.joystick.get_axis(1)  # Inversion pour l'intuition

        pos_x = x + int(steering * size//2 * 0.95)  # 0.95 pour éviter de toucher le bord
        pos_y = y - int(accel * size//2 * 0.95)

        # Trajet du centre à la position (avec dégradé)
        steps = 10
        for i in range(steps):
            prog = i / steps
            step_x = x + int(steering * size//2 * 0.95 * prog)
            step_y = y - int(accel * size//2 * 0.95 * prog)
            alpha = int(200 * prog)
            color = (self.RED[0], self.RED[1], self.RED[2], alpha)
            pygame.draw.circle(
                self.window,
                color,
                (step_x, step_y),
                2 + i//3
            )

        # Point représentant la position actuelle (plus grand et plus visible)
        glow_size = 8 + int(4 * self.pulse_value)
        # Halo extérieur
        pygame.draw.circle(self.window, self.RED, (pos_x, pos_y), glow_size)
        # Cercle intérieur
        pygame.draw.circle(self.window, self.WHITE, (pos_x, pos_y), glow_size - 4)

    def draw_interface(self, elapsed_time):
        """Dessine l'interface complète de calibration avec un style amélioré."""
        # Fond dégradé
        self.draw_gradient_background()

        # Cadre principal
        margin = 20
        main_rect = (margin, margin, self.width - 2*margin, self.height - 2*margin)
        self.draw_rounded_rect(self.window, main_rect, (*self.BACKGROUND_TOP, 150))

        # Titre avec effet ombre
        title_y = 50
        self.draw_text_with_shadow(
            "Calibration du Joystick",
            self.font,
            self.WHITE,
            (self.width//2 - self.font.size("Calibration du Joystick")[0]//2, title_y)
        )

        # Barre de progression pour le temps
        progress = min(1.0, elapsed_time / self.calibration_duration)
        progress_width = self.width - 100
        self.draw_progress_bar(
            50, title_y + 50,
            progress_width, 20,
            progress,
            self.DARK_GRAY,
            self.TEAL,
            self.LIGHT_GRAY
        )

        # Texte du temps restant
        time_left = max(0, self.calibration_duration - elapsed_time)
        time_text = f"Temps restant: {time_left:.1f}s"
        time_text_pos = (self.width//2 - self.small_font.size(time_text)[0]//2, title_y + 80)
        self.draw_text_with_shadow(time_text, self.small_font, self.WHITE, time_text_pos)

        # Instructions
        instructions = [
            "Déplacez le joystick dans toutes les directions extrêmes",
            "Appuyez sur ENTRÉE pour terminer ou ÉCHAP pour annuler"
        ]

        for i, text in enumerate(instructions):
            y_pos = title_y + 120 + i * 30
            self.draw_text_with_shadow(
                text,
                self.small_font,
                self.LIGHT_GRAY,
                (self.width//2 - self.small_font.size(text)[0]//2, y_pos)
            )

        # Visualisation des axes
        self.draw_axis_visualization(
            80, 250, self.width - 160, 40,
            self.joystick.get_axis(0),
            self.steering_min, self.steering_max,
            "Direction", self.BLUE
        )

        self.draw_axis_visualization(
            80, 350, self.width - 160, 40,
            self.joystick.get_axis(1),
            self.accel_min, self.accel_max,
            "Accélération", self.RED
        )

        # Visualisation du joystick
        self.draw_joystick_position(self.width//2, 480, 200)

        # Informations de commandes
        commands = [
            "ENTRÉE: Sauvegarder et quitter",
            "ÉCHAP: Annuler et quitter"
        ]

        for i, cmd in enumerate(commands):
            cmd_y = self.height - 50 + i * 20
            self.draw_text_with_shadow(
                cmd,
                self.tiny_font,
                self.LIGHT_GRAY,
                (self.width - self.tiny_font.size(cmd)[0] - 30, cmd_y)
            )

    def check_events(self):
        """
        Vérifie les événements utilisateur.

        Returns:
            bool: True si la calibration doit continuer, False sinon
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_RETURN:
                    self.save_calibration()
                    return False
        return True

    def run(self):
        """Exécute la boucle principale de l'interface de calibration."""
        if not self.joystick:
            return False

        clock = pygame.time.Clock()

        while self.is_calibrating:
            # Gestion des événements
            self.is_calibrating = self.check_events()

            # Mise à jour des valeurs de calibration
            self.update_calibration_values()

            # Vérification du temps écoulé
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.calibration_duration:
                self.save_calibration()
                self.is_calibrating = False

            # Dessin de l'interface
            self.draw_interface(elapsed_time)

            # Mise à jour de l'affichage
            pygame.display.flip()
            clock.tick(30)

        # Nettoyage - ne pas fermer la fenêtre avec set_mode mais avec flip et delay
        if pygame.display.get_init():
            pygame.display.flip()
            pygame.time.delay(100)  # Délai pour permettre à pygame de traiter

        return True


def calibrate_joystick(joystick):
    """
    Lance l'interface de calibration pour un joystick.

    Args:
        joystick: Instance du joystick Pygame à calibrer

    Returns:
        bool: True si la calibration a été effectuée, False sinon
    """
    if not joystick:
        print("[ERREUR] Aucun joystick à calibrer")
        return False

    # Sauvegarder l'ID du joystick pour vérification
    try:
        joystick_name = joystick.get_name()
        joystick_id = joystick.get_id()
    except (pygame.error, AttributeError):
        print("[AVERTISSEMENT] Impossible d'obtenir les infos du joystick")
        joystick_name = "Unknown"
        joystick_id = -1

    # Lancer la calibration
    calibrator = JoystickCalibratorUI(joystick)
    result = calibrator.run()

    print("[INFO] Calibration terminée, nettoyage des événements pygame...")

    # S'assurer que pygame est toujours initialisé
    if not pygame.get_init():
        pygame.init()

    # Vider complètement la file d'attente d'événements
    pygame.event.clear()

    # Laisser pygame gérer les événements
    for _ in range(20):
        pygame.event.pump()
        pygame.time.delay(10)

    return result
