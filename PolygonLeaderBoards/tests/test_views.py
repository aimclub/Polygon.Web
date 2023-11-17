from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth import get_user

import os
import shutil


class TestViews(TestCase):

	def setUp(self):
		self.client = Client()
		self.home_url = reverse('HomePage')
		self.lboards_url = reverse('LBoardsPage')
		self.upload_url = reverse('UploadPage')
		self.user_url = reverse('UserPage')
		self.login_url = reverse('LoginPage')


	def test_HomePage(self):

		response = self.client.get(self.home_url)

		self.assertEqual(response.status_code, 200)
		self.assertTemplateUsed(response, 'html/home.html')


	def test_LBoardsPage(self):

		response = self.client.get(self.lboards_url)

		self.assertEqual(response.status_code, 200)
		self.assertTemplateUsed(response, 'html/learderboards.html')


	def test_UploadPage(self):

		response = self.client.get(self.upload_url)

		self.assertEqual(response.status_code, 200)
		self.assertTemplateUsed(response, 'html/upload.html')
